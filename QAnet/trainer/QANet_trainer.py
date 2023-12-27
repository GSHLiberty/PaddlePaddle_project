import sys
sys.path.append('/home/gsh/paddle_project/QAnet/utils')
# from utils import paddle_aux
import paddle
"""
Trainer file for SQuAD dataset.
"""
import os
import shutil
import time
from datetime import datetime
from .metric import convert_tokens, evaluate_by_dict
from util.file_utils import pickle_load_large_file


class Trainer(object):

    def __init__(self, args, model, loss, train_data_loader,
        dev_data_loader, train_eval_file, dev_eval_file, optimizer,
        scheduler, epochs, with_cuda, save_dir, verbosity=2, save_freq=1,
        print_freq=10, resume=False, identifier='', debug=False,
        debug_batchnum=2, visualizer=None, logger=None, grad_clip=5.0,
        decay=0.9999, lr=0.001, lr_warm_up_num=1000, use_scheduler=False,
        use_grad_clip=False, use_ema=False, ema=None, use_early_stop=False,
        early_stop=10):
        self.place = str('gpu' if with_cuda else 'cpu')
        self.args = args
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.verbosity = verbosity
        self.identifier = identifier
        self.visualizer = visualizer
        self.with_cuda = with_cuda
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.dev_eval_dict = pickle_load_large_file(dev_eval_file)
        self.is_debug = debug
        self.debug_batchnum = debug_batchnum
        self.logger = logger
        self.unused = True
        self.lr = lr
        self.lr_warm_up_num = lr_warm_up_num
        self.decay = decay
        self.use_scheduler = use_scheduler
        self.scheduler = scheduler
        self.use_grad_clip = use_grad_clip
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        self.ema = ema
        self.use_early_stop = use_early_stop
        self.early_stop = early_stop
        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        self.start_epoch = 1
        self.step = 0
        self.best_em = 0
        self.best_f1 = 0
        if resume:
            self._resume_checkpoint(resume)
            self.model = self.model.to(self.place)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, paddle.Tensor):
                        state[k] = v.to(self.place)

    def train(self):
        patience = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            if self.use_early_stop:
                if result['f1'] < self.best_f1 and result['em'] < self.best_em:
                    patience += 1
                    if patience > self.early_stop:
                        print('Perform early stop!')
                        break
                else:
                    patience = 0
            is_best = False
            if result['f1'] > self.best_f1:
                is_best = True
            if result['f1'] == self.best_f1 and result['em'] > self.best_em:
                is_best = True
            self.best_f1 = max(self.best_f1, result['f1'])
            self.best_em = max(self.best_em, result['em'])
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, result['f1'], result['em'],
                    is_best)

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.place)
        global_loss = 0.0
        last_step = self.step - 1
        last_time = time.time()
        for batch_idx, batch in enumerate(self.train_data_loader):
            (context_wids, context_cids, question_wids, question_cids, y1,
                y2, y1s, y2s, id, answerable) = batch
            batch_num, question_len = question_wids.shape
            _, context_len = context_wids.shape
            context_wids = paddle.to_tensor(context_wids, place=self.place)
            context_cids = paddle.to_tensor(context_cids, place=self.place)
            question_wids = paddle.to_tensor(question_wids, place=self.place)
            question_cids = paddle.to_tensor(question_cids, place=self.place)
            y1 = paddle.to_tensor(y1, place=self.place)
            y2 = paddle.to_tensor(y2, place=self.place)
            id = paddle.to_tensor(id, place=self.place)
            answerable = paddle.to_tensor(answerable, place=self.place)
            self.model.clear_gradients()
            p1, p2 = self.model(context_wids, context_cids, question_wids,
                question_cids)
            loss1 = self.loss(p1, y1)
            loss2 = self.loss(p2, y2)
            loss = paddle.mean(x=loss1 + loss2)
            loss.backward()
            global_loss += loss.item()
            self.optimizer.step()
            if self.use_scheduler:
                self.scheduler.step()
            if self.use_ema and self.ema is not None:
                self.ema(self.model, self.step)
            if self.step % self.print_freq == self.print_freq - 1:
                used_time = time.time() - last_time
                step_num = self.step - last_step
                speed = (self.train_data_loader.batch_size * step_num /
                    used_time)
                batch_loss = global_loss / step_num
                print(
                    'step: {}/{} \t epoch: {} \t lr: {} \t loss: {} \t speed: {} examples/sec'
                    .format(batch_idx, len(self.train_data_loader), epoch,
                    self.scheduler.get_lr(), batch_loss, speed))
                global_loss = 0.0
                last_step = self.step
                last_time = time.time()
            self.step += 1
            if self.is_debug and batch_idx >= self.debug_batchnum:
                break
        metrics = self._valid_eopch(self.dev_eval_dict, self.dev_data_loader)
        print('dev_em: %f \t dev_f1: %f' % (metrics['exact_match'], metrics
            ['f1']))
        result = {}
        result['em'] = metrics['exact_match']
        result['f1'] = metrics['f1']
        return result

    def _valid_eopch(self, eval_dict, data_loader):
        """
        Evaluate model over development dataset.
        Return the metrics: em, f1.
        """
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)
        self.model.eval()
        answer_dict = {}
        with paddle.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                (context_wids, context_cids, question_wids, question_cids,
                    y1, y2, y1s, y2s, id, answerable) = batch
                context_wids = paddle.to_tensor(context_wids, place=self.place)
                context_cids = paddle.to_tensor(context_cids, place=self.place)
                question_wids = paddle.to_tensor(question_wids, place=self.place)
                question_cids = paddle.to_tensor(question_cids, place=self.place)
                y1 = paddle.to_tensor(y1, place=self.place)
                y2 = paddle.to_tensor(y2, place=self.place)
                answerable = paddle.to_tensor(answerable, place=self.place)
                p1, p2 = self.model(context_wids, context_cids,
                    question_wids, question_cids)
                p1 = paddle.nn.functional.softmax(x=p1, axis=1)
                p2 = paddle.nn.functional.softmax(x=p2, axis=1)
                outer = paddle.matmul(x=p1.unsqueeze(axis=2), y=p2.
                    unsqueeze(axis=1))
                for j in range(outer.shape[0]):
                    outer[j] = paddle.triu(x=outer[j])
                a1, _ = paddle.max(x=outer, axis=2), paddle.argmax(x=outer,
                    axis=2)
                a2, _ = paddle.max(x=outer, axis=1), paddle.argmax(x=outer,
                    axis=1)
                ymin = paddle.argmax(x=a1, axis=1)
                ymax = paddle.argmax(x=a2, axis=1)
                answer_dict_, _ = convert_tokens(eval_dict, id.tolist(),
                    ymin.tolist(), ymax.tolist())
                answer_dict.update(answer_dict_)
                if batch_idx + 1 == self.args.val_num_batches:
                    break
                if self.is_debug and batch_idx >= self.debug_batchnum:
                    break
        metrics = evaluate_by_dict(eval_dict, answer_dict)
        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch, f1, em, is_best):
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)
        arch = type(self.model).__name__
        param_path = os.path.join(self.save_dir, self.identifier + 'checkpoint_epoch{:02d}_f1_{:.5f}_em_{:.5f}.pdparams'.format(epoch, f1, em))
        paddle.save(self.model.state_dict(), param_path)
        print('Saving checkpoint: {} ...'.format(param_path))
        opt_path = os.path.join(self.save_dir, self.identifier + 'checkpoint_epoch{:02d}_f1_{:.5f}_em_{:.5f}.pdopt'.format(epoch, f1, em))
        paddle.save(self.optimizer.state_dict(), opt_path)
        print('Saving checkpoint: {} ...'.format(opt_path))
        if is_best:
            shutil.copyfile(param_path, os.path.join(self.save_dir, 'model_best.pdparams'))
            shutil.copyfile(opt_path, os.path.join(self.save_dir, 'model_best.pdopt'))
        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        return param_path

    def _resume_checkpoint(self, resume_path):
        print('Loading checkpoint: {} ...'.format(resume_path))
        checkpoint = paddle.load(path=resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_f1 = checkpoint['best_f1']
        self.best_em = checkpoint['best_em']
        self.step = checkpoint['step']
        self.start_time = checkpoint['start_time']
        if self.use_scheduler:
            self.scheduler.last_epoch = checkpoint['epoch']
        print("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.
            start_epoch))
