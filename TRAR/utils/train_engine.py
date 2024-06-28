import paddle
import os, datetime, shutil, time
import numpy as np
from openvqa.models.model_loader import ModelLoader
from openvqa.utils.optim import get_optim, adjust_lr
from utils.test_engine import test_engine, ckpt_proc


def set_training_tau(__C, net, tau):
    if __C.N_GPU > 1:
        net.module.backbone.set_tau(tau)
    else:
        net.backbone.set_tau(tau)
    return tau


def train_engine(__C, dataset, dataset_eval=None):
    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb
    net = ModelLoader(__C).Net(__C, pretrained_emb, token_size, ans_size)
    net.train()
    if __C.N_GPU > 1:
        net = paddle.DataParallel(net)
    loss_fn = eval('paddle.nn.' + __C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] +
        "(reduction='" + __C.LOSS_REDUCTION + "')")
    if __C.RESUME:
        print(' ========== Resume training')
        if __C.CKPT_PATH is not None:
            print(
                'Warning: Now using CKPT_PATH args, CKPT_VERSION and CKPT_EPOCH will not work'
                )
            path = __C.CKPT_PATH
        else:
            path = (__C.CKPTS_PATH + '/ckpt_' + __C.CKPT_VERSION + '/epoch' +
                str(__C.CKPT_EPOCH) + '.pkl')
        print('Loading ckpt from {}'.format(path))
        ckpt = paddle.load(path=path)
        print('Finish!')
        if __C.N_GPU > 1:
            net.set_state_dict(state_dict=ckpt_proc(ckpt['state_dict']))
        else:
            net.set_state_dict(state_dict=ckpt['state_dict'])
        start_epoch = ckpt['epoch']
        optim = get_optim(__C, net, data_size, ckpt['lr_base'])
        optim._step = int(data_size / __C.BATCH_SIZE * start_epoch)
        optim.optimizer.load_state_dict(ckpt['optimizer'])
        if 'ckpt_' + __C.VERSION not in os.listdir(__C.CKPTS_PATH):
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)
    else:
        if 'ckpt_' + __C.VERSION not in os.listdir(__C.CKPTS_PATH):
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)
        optim = get_optim(__C, net, data_size)
        start_epoch = 0
    loss_sum = 0
    named_params = list(net.named_parameters())
    grad_norm = np.zeros(len(named_params))
    dataloader = paddle.io.DataLoader(dataset, batch_size=__C.BATCH_SIZE, shuffle=True, num_workers=__C.NUM_WORKERS, drop_last=True)
    logfile = open(__C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt', 'a+')
    logfile.write(str(__C))
    logfile.close()
    for epoch in range(start_epoch, __C.MAX_EPOCH):
        logfile = open(__C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt', 'a+')
        logfile.write('=====================================\nnowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        logfile.close()
        if epoch in __C.LR_DECAY_LIST:
            adjust_lr(optim, __C.LR_DECAY_R)
        if __C.ROUTING == 'hard':
            print('setting training temperature...')
            if __C.TAU_POLICY == 0:
                tau = __C.TAU_MAX - (__C.TAU_MAX - __C.TAU_MIN) * epoch / (__C
                    .MAX_EPOCH - 1)
                set_training_tau(__C, net, tau)
            elif __C.TAU_POLICY == 1:
                if epoch < __C.WARMUP_EPOCH:
                    tau = __C.TAU_MAX - (__C.TAU_MAX - 1
                        ) * epoch / __C.WARMUP_EPOCH
                    set_training_tau(__C, net, tau)
                else:
                    tau = 1.0 - (epoch - __C.WARMUP_EPOCH) / (__C.MAX_EPOCH -
                        __C.WARMUP_EPOCH)
                    set_training_tau(__C, net, tau)
            elif __C.TAU_POLICY == 2:
                if epoch < __C.WARMUP_EPOCH:
                    tau = 1.0 - (epoch - __C.MAX_EPOCH) / (__C.WARMUP_EPOCH + 1
                        )
                    set_training_tau(__C, net, tau)
                else:
                    tau = 0.1
                    set_training_tau(__C, net, tau)
            print('epoch %d: setting training temperature to %2.5f' % (
                epoch, tau))
        elif __C.ROUTING == 'soft':
            print('using soft routing block')
        time_start = time.time()
        for step, (frcn_feat_iter, grid_feat_iter, bbox_feat_iter,
            ques_ix_iter, ans_iter) in enumerate(dataloader):
            optim.zero_grad()
            frcn_feat_iter = frcn_feat_iter
            grid_feat_iter = grid_feat_iter
            bbox_feat_iter = bbox_feat_iter
            ques_ix_iter = ques_ix_iter
            ans_iter = ans_iter
            loss_tmp = 0
            for accu_step in range(__C.GRAD_ACCU_STEPS):
                loss_tmp = 0
                sub_frcn_feat_iter = frcn_feat_iter[accu_step * __C.
                    SUB_BATCH_SIZE:(accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_grid_feat_iter = grid_feat_iter[accu_step * __C.
                    SUB_BATCH_SIZE:(accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_bbox_feat_iter = bbox_feat_iter[accu_step * __C.
                    SUB_BATCH_SIZE:(accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ques_ix_iter = ques_ix_iter[accu_step * __C.
                    SUB_BATCH_SIZE:(accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ans_iter = ans_iter[accu_step * __C.SUB_BATCH_SIZE:(
                    accu_step + 1) * __C.SUB_BATCH_SIZE]
                pred = net(sub_frcn_feat_iter, sub_grid_feat_iter,
                    sub_bbox_feat_iter, sub_ques_ix_iter)
                loss_item = [pred, sub_ans_iter]
                loss_nonlinear_list = __C.LOSS_FUNC_NONLINEAR[__C.LOSS_FUNC]
                for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                    if loss_nonlinear in ['flat']:
                        loss_item[item_ix] = paddle.reshape(loss_item[item_ix], shape=[-1])
                    elif loss_nonlinear:
                        loss_item[item_ix] = eval('F.' + loss_nonlinear +
                            '(loss_item[item_ix], dim=1)')
                loss = loss_fn(loss_item[0], loss_item[1])
                if __C.LOSS_REDUCTION == 'mean':
                    loss /= __C.GRAD_ACCU_STEPS
                loss.backward()
                loss_tmp += loss.cpu().numpy() * __C.GRAD_ACCU_STEPS
                loss_sum += loss.cpu().numpy() * __C.GRAD_ACCU_STEPS
            if __C.VERBOSE:
                if dataset_eval is not None:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['val']
                else:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['test']
                print(
                    '\r[Version %s][Model %s][Dataset %s][Epoch %2d][Step %4d/%4d][%s] Loss: %.4f, Lr: %.2e'
                     % (__C.VERSION, __C.MODEL_USE, __C.DATASET, epoch + 1,
                    step, int(data_size / __C.BATCH_SIZE), mode_str, 
                    loss_tmp / __C.SUB_BATCH_SIZE, optim._rate), end=
                    '          ')
            if __C.GRAD_NORM_CLIP > 0:
                params_to_clip = [p for p in net.parameters() if len(p.shape) > 0]
                paddle.nn.utils.clip_grad_norm_(parameters=params_to_clip, max_norm=__C.GRAD_NORM_CLIP)
            for name in range(len(named_params)):
                if named_params[name][1].grad is not None:
                    norm_v = paddle.norm(named_params[name][1].grad).numpy()
                else:
                    norm_v = 0

                grad_norm[name] += norm_v * __C.GRAD_ACCU_STEPS
            optim.step()
        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1
        if __C.N_GPU > 1:
            state = {'state_dict': net.module.state_dict(), 'optimizer':
                optim.optimizer.state_dict(), 'lr_base': optim.lr_base,
                'epoch': epoch_finish}
        else:
            state = {'state_dict': net.state_dict(), 'optimizer': optim.
                optimizer.state_dict(), 'lr_base': optim.lr_base, 'epoch':
                epoch_finish}
        paddle.save(obj=state, path=__C.CKPTS_PATH + '/ckpt_' + __C.VERSION +
            '/epoch' + str(epoch_finish) + '.pkl')
        logfile = open(__C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt', 'a+')
        logfile.write('Epoch: ' + str(epoch_finish) + ', Loss: ' + str(
            loss_sum / data_size) + ', Lr: ' + str(optim._rate) + '\n' +
            'Elapsed time: ' + str(int(elapse_time))  + '\n\n')
        logfile.close()
        if dataset_eval is not None:
            net.backbone.set_training_status(False)
            test_engine(__C, dataset_eval, state_dict=net.state_dict(),
                validation=True)
            net.backbone.set_training_status(True)
        loss_sum = 0
        grad_norm = np.zeros(len(named_params))
