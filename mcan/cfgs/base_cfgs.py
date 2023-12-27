import paddle
from cfgs.path_cfgs import PATH
import os, random
import numpy as np
from types import MethodType


class Cfgs(PATH):

    def __init__(self):
        super(Cfgs, self).__init__()
        self.GPU = '0'
        self.SEED = random.randint(0, 99999999)
        self.VERSION = str(self.SEED)
        self.RESUME = False
        self.CKPT_VERSION = self.VERSION
        self.CKPT_EPOCH = 0
        self.CKPT_PATH = None
        self.VERBOSE = True
        self.RUN_MODE = 'train'
        self.EVAL_EVERY_EPOCH = True
        self.TEST_SAVE_PRED = False
        self.PRELOAD = False
        self.SPLIT = {'train': '', 'val': 'val', 'test': 'test'}
        self.TRAIN_SPLIT = 'train+val+vg'
        self.USE_GLOVE = True
        self.WORD_EMBED_SIZE = 300
        self.MAX_TOKEN = 14
        self.IMG_FEAT_PAD_SIZE = 100
        self.IMG_FEAT_SIZE = 2048
        self.BATCH_SIZE = 64
        # self.NUM_WORKERS = 8
        self.NUM_WORKERS = 0
        self.PIN_MEM = True
        self.GRAD_ACCU_STEPS = 1
        self.SHUFFLE_MODE = 'external'
        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.LR_BASE = 0.0001
        self.LR_DECAY_R = 0.2
        self.LR_DECAY_LIST = [10, 12]
        self.MAX_EPOCH = 13
        self.GRAD_NORM_CLIP = -1
        self.OPT_BETAS = 0.9, 0.98
        self.OPT_EPS = 1e-09

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg
                ), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def proc(self):
        assert self.RUN_MODE in ['train', 'val', 'test']
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
#         paddle.set_num_threads(2)
# >>>>>>        torch.set_num_threads(2)
        paddle.seed(seed=self.SEED)
        if self.N_GPU < 2:
            paddle.seed(seed=self.SEED)
        else:
            paddle.seed(seed=self.SEED)
        paddle.framework.set_flags({'FLAGS_cudnn_deterministic': True})
        # False = True
        np.random.seed(self.SEED)
        random.seed(self.SEED)
        if self.CKPT_PATH is not None:
            print(
                'Warning: you are now using CKPT_PATH args, CKPT_VERSION and CKPT_EPOCH will not work'
                )
            self.CKPT_VERSION = self.CKPT_PATH.split('/')[-1] + '_' + str(
                random.randint(0, 99999999))
        self.SPLIT['train'] = self.TRAIN_SPLIT
        if 'val' in self.SPLIT['train'].split('+') or self.RUN_MODE not in ['train']:
            self.EVAL_EVERY_EPOCH = False
        if self.RUN_MODE not in ['test']:
            self.TEST_SAVE_PRED = False
        assert self.BATCH_SIZE % self.GRAD_ACCU_STEPS == 0
        self.SUB_BATCH_SIZE = int(self.BATCH_SIZE / self.GRAD_ACCU_STEPS)
        self.EVAL_BATCH_SIZE = int(self.SUB_BATCH_SIZE / 2)
        self.FF_SIZE = int(self.HIDDEN_SIZE * 4)
        assert self.HIDDEN_SIZE % self.MULTI_HEAD == 0
        self.HIDDEN_SIZE_HEAD = int(self.HIDDEN_SIZE / self.MULTI_HEAD)

    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self,
                attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))
        return ''
