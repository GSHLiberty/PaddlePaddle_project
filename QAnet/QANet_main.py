import sys
from utils import paddle_aux
import paddle
import paddle.optimizer
"""
Main file for training SQuAD reading comprehension model.
"""
import os
import sys
import argparse
import math
import random
import numpy as np
from datetime import datetime
from data_loader.SQuAD import prepro, get_loader
from model.QANet import QANet
from trainer.QANet_trainer import Trainer
from util.visualize import Visualizer
from model.modules.ema import EMA
from util.file_utils import pickle_load_large_file
data_folder = '/home/gsh/paddle_project/QAnet/datasets/'
parser = argparse.ArgumentParser(description='Lucy')
parser.add_argument('--processed_data', default=False, action='store_true',
    help='whether the dataset already processed')
parser.add_argument('--train_file', default=data_folder +
    'original/SQuAD/train-v1.1.json', type=str, help='path of train dataset')
parser.add_argument('--dev_file', default=data_folder +
    'original/SQuAD/dev-v1.1.json', type=str, help='path of dev dataset')
parser.add_argument('--train_examples_file', default=data_folder +
    'processed/SQuAD/train-v1.1-examples.pkl', type=str, help=
    'path of train dataset examples file')
parser.add_argument('--dev_examples_file', default=data_folder +
    'processed/SQuAD/dev-v1.1-examples.pkl', type=str, help=
    'path of dev dataset examples file')
parser.add_argument('--train_meta_file', default=data_folder +
    'processed/SQuAD/train-v1.1-meta.pkl', type=str, help=
    'path of train dataset meta file')
parser.add_argument('--dev_meta_file', default=data_folder +
    'processed/SQuAD/dev-v1.1-meta.pkl', type=str, help=
    'path of dev dataset meta file')
parser.add_argument('--train_eval_file', default=data_folder +
    'processed/SQuAD/train-v1.1-eval.pkl', type=str, help=
    'path of train dataset eval file')
parser.add_argument('--dev_eval_file', default=data_folder +
    'processed/SQuAD/dev-v1.1-eval.pkl', type=str, help=
    'path of dev dataset eval file')
parser.add_argument('--val_num_batches', default=500, type=int, help=
    'number of batches for evaluation (default: 500)')
parser.add_argument('--glove_word_file', default=data_folder +
    'original/Glove/glove.840B.300d.txt', type=str, help=
    'path of word embedding file')
parser.add_argument('--glove_word_size', default=int(2200000.0), type=int,
    help='Corpus size for Glove')
parser.add_argument('--glove_dim', default=300, type=int, help=
    'word embedding size (default: 300)')
parser.add_argument('--word_emb_file', default=data_folder +
    'processed/SQuAD/word_emb.pkl', type=str, help=
    'path of word embedding matrix file')
parser.add_argument('--word_dictionary', default=data_folder +
    'processed/SQuAD/word_dict.pkl', type=str, help=
    'path of word embedding dict file')
parser.add_argument('--pretrained_char', default=False, action='store_true',
    help='whether train char embedding or not')
parser.add_argument('--glove_char_file', default=data_folder +
    'original/Glove/glove.840B.300d-char.txt', type=str, help=
    'path of char embedding file')
parser.add_argument('--glove_char_size', default=94, type=int, help=
    'Corpus size for char embedding')
parser.add_argument('--char_dim', default=64, type=int, help=
    'char embedding size (default: 64)')
parser.add_argument('--char_emb_file', default=data_folder +
    'processed/SQuAD/char_emb.pkl', type=str, help=
    'path of char embedding matrix file')
parser.add_argument('--char_dictionary', default=data_folder +
    'processed/SQuAD/char_dict.pkl', type=str, help=
    'path of char embedding dict file')
parser.add_argument('-b', '--batch_size', default=32, type=int, help=
    'mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=30, type=int, help=
    'number of total epochs (default: 30)')
parser.add_argument('--debug', default=False, action='store_true', help=
    'debug mode or not')
parser.add_argument('--debug_batchnum', default=2, type=int, help=
    'only train and test a few batches when debug (devault: 2)')
parser.add_argument('--resume', default='', type=str, help=
    'path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=2, type=int, help=
    'verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save_dir', default='checkpoints/', type=str, help=
    'directory of saved model (default: checkpoints/)')
parser.add_argument('--save_freq', default=1, type=int, help=
    'training checkpoint frequency (default: 1 epoch)')
parser.add_argument('--print_freq', default=10, type=int, help=
    'print training information frequency (default: 10 steps)')
parser.add_argument('--with_cuda', default=False, action='store_true', help
    ="use CPU in case there's no GPU support")
parser.add_argument('--multi_gpu', default=False, action='store_true', help
    ="use multi-GPU in case there's multiple GPUs available")
parser.add_argument('--visualizer', default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument('--log_file', default='log.txt', type=str, help=
    'path of log file')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lr_warm_up_num', default=1000, type=int, help=
    'number of warm-up steps of learning rate')
parser.add_argument('--beta1', default=0.8, type=float, help='beta 1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta 2')
parser.add_argument('--decay', default=0.9999, type=float, help=
    'exponential moving average decay')
parser.add_argument('--use_scheduler', default=True, action='store_false',
    help='whether use learning rate scheduler')
parser.add_argument('--use_grad_clip', default=True, action='store_false',
    help='whether use gradient clip')
parser.add_argument('--grad_clip', default=5.0, type=float, help=
    'global Norm gradient clipping rate')
parser.add_argument('--use_ema', default=False, action='store_true', help=
    'whether use exponential moving average')
parser.add_argument('--use_early_stop', default=True, action='store_false',
    help='whether use early stop')
parser.add_argument('--early_stop', default=10, type=int, help=
    'checkpoints for early stop')
parser.add_argument('--para_limit', default=400, type=int, help=
    'maximum context token number')
parser.add_argument('--ques_limit', default=50, type=int, help=
    'maximum question token number')
parser.add_argument('--ans_limit', default=30, type=int, help=
    'maximum answer token number')
parser.add_argument('--char_limit', default=16, type=int, help=
    'maximum char number in a word')
parser.add_argument('--d_model', default=128, type=int, help=
    'model hidden size')
parser.add_argument('--num_head', default=8, type=int, help=
    'attention num head')


def main(args):
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    random_seed = None
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        paddle.seed(seed=random_seed)
    log = sys.stdout
    if args.log_file is not None:
        log = open(args.log_file, 'a')
    device = str('cuda:2' if paddle.device.cuda.device_count() >= 1 else 'cpu'
        ).replace('cuda', 'gpu')
    n_gpu = paddle.device.cuda.device_count()
    if paddle.device.cuda.device_count() >= 1:
        print('device is cuda, # cuda is: ', n_gpu)
    else:
        print('device is cpu')
    if not args.processed_data:
        prepro(args)
    wv_tensor = paddle.to_tensor(data=np.array(pickle_load_large_file(args.
        word_emb_file), dtype=np.float32), dtype='float32')
    cv_tensor = paddle.to_tensor(data=np.array(pickle_load_large_file(args.
        char_emb_file), dtype=np.float32), dtype='float32')
    wv_word2ix = pickle_load_large_file(args.word_dictionary)
    train_dataloader = get_loader(args.train_examples_file, args.batch_size,
        shuffle=True)
    dev_dataloader = get_loader(args.dev_examples_file, args.batch_size,
        shuffle=True)
    model = QANet(wv_tensor, cv_tensor, args.para_limit, args.ques_limit,
        args.d_model, num_head=args.num_head, train_cemb=not args.
        pretrained_char, pad=wv_word2ix['<PAD>'])
    model.summary()
    if paddle.device.cuda.device_count() > 1 and args.multi_gpu:
        model = paddle.DataParallel(model)
    model.to(device)
    ema = EMA(args.decay)
    if args.use_ema:
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                ema.register(name, param)
    parameters = filter(lambda p: not p.stop_gradient, model.parameters())
    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.Adam(parameters=parameters,
                    learning_rate=args.lr,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    epsilon=1e-08,
                    weight_decay=3e-7,
                    grad_clip=clip)
    
    cr = 1.0 / math.log(args.lr_warm_up_num)
    def lr_lambda(step):
        return cr * math.log(step + 1) if step < args.lr_warm_up_num else 1.0

    # Create the LambdaDecay learning rate scheduler
    scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate=optimizer.get_lr(), lr_lambda=lr_lambda)
    loss = paddle.nn.CrossEntropyLoss()
    vis = None
    if args.visualizer:
        os.system('python -m visdom.server')
        vis = Visualizer('main')
    identifier = type(model).__name__ + '_'
    trainer = Trainer(args, model, loss, train_data_loader=train_dataloader,
        dev_data_loader=dev_dataloader, train_eval_file=args.
        train_eval_file, dev_eval_file=args.dev_eval_file, optimizer=
        optimizer, scheduler=scheduler, epochs=args.epochs, with_cuda=args.
        with_cuda, save_dir=args.save_dir, verbosity=args.verbosity,
        save_freq=args.save_freq, print_freq=args.print_freq, resume=args.
        resume, identifier=identifier, debug=args.debug, debug_batchnum=
        args.debug_batchnum, lr=args.lr, lr_warm_up_num=args.lr_warm_up_num,
        grad_clip=args.grad_clip, decay=args.decay, visualizer=vis, logger=
        log, use_scheduler=args.use_scheduler, use_grad_clip=args.
        use_grad_clip, use_ema=args.use_ema, ema=ema, use_early_stop=args.
        use_early_stop, early_stop=args.early_stop)
    start = datetime.now()
    trainer.train()
    print('Time of training model ', datetime.now() - start)


if __name__ == '__main__':
    main(parser.parse_args())
