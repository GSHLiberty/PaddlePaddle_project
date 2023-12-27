import paddle
from openvqa.core.path_cfgs import PATH
import os, random
import numpy as np
from types import MethodType


class BaseCfgs(PATH):

    def __init__(self):
        super(BaseCfgs, self).__init__()
        self.GPU = '0'
        self.SEED = random.randint(0, 9999999)
        self.VERSION = str(self.SEED)
        self.RESUME = False
        self.CKPT_VERSION = self.VERSION
        self.CKPT_EPOCH = 0
        self.CKPT_PATH = None
        self.VERBOSE = True
        self.MODEL = ''
        self.MODEL_USE = ''
        self.DATASET = ''
        self.RUN_MODE = ''
        self.EVAL_EVERY_EPOCH = True
        self.TEST_SAVE_PRED = False
        self.TRAIN_SPLIT = 'train'
        self.USE_GLOVE = True
        self.WORD_EMBED_SIZE = 300
        self.FEAT_SIZE = {'vqa': {'FRCN_FEAT_SIZE': (100, 2048),
            'BBOX_FEAT_SIZE': (100, 5)}, 'gqa': {'FRCN_FEAT_SIZE': (100, 
            2048), 'GRID_FEAT_SIZE': (49, 2048), 'BBOX_FEAT_SIZE': (100, 5)
            }, 'clevr': {'GRID_FEAT_SIZE': (196, 1024)}}
        self.BBOX_NORMALIZE = False
        self.BATCH_SIZE = 64
        self.NUM_WORKERS = 0
        self.PIN_MEM = True
        self.GRAD_ACCU_STEPS = 1
        """
        Loss(case-sensitive): 
        'ce'    : Cross Entropy -> NLLLoss(LogSoftmax(output), label) = CrossEntropyLoss(output, label)
        'bce'   : Binary Cross Entropy -> BCELoss(Sigmoid(output), label) = BCEWithLogitsLoss(output, label)
        'kld'   : Kullback-Leibler Divergence -> KLDivLoss(LogSoftmax(output), Softmax(label))
        'mse'   : Mean Squared Error -> MSELoss(output, label)
        
        Reduction(case-sensitive):
        'none': no reduction will be applied
        'elementwise_mean': the sum of the output will be divided by the number of elements in the output
        'sum': the output will be summed
        """
        self.LOSS_FUNC = ''
        self.LOSS_REDUCTION = ''
        self.LR_BASE = 0.0001
        self.LR_DECAY_R = 0.2
        self.LR_DECAY_LIST = [10, 12]
        self.WARMUP_EPOCH = 3
        self.MAX_EPOCH = 13
        self.GRAD_NORM_CLIP = -1
        """
        Optimizer(case-sensitive): 
        'Adam'      : default -> {betas:(0.9, 0.999), eps:1e-8, weight_decay:0, amsgrad:False}
        'Adamax'    : default -> {betas:(0.9, 0.999), eps:1e-8, weight_decay:0}
        'RMSprop'   : default -> {alpha:0.99, eps:1e-8, weight_decay:0, momentum:0, centered:False}
        'SGD'       : default -> {momentum:0, dampening:0, weight_decay:0, nesterov:False}
        'Adadelta'  : default -> {rho:0.9, eps:1e-6, weight_decay:0}
        'Adagrad'   : default -> {lr_decay:0, weight_decay:0, initial_accumulator_value:0}
        
        In YML files:
        If you want to self-define the optimizer parameters, set a dict named OPT_PARAMS contains the keys you want to modify.
         !!! Warning: keys: ['params, 'lr'] should not be set. 
         !!! Warning: To avoid ambiguity, the value of keys should be defined as string type.
        If you not define the OPT_PARAMS, all parameters of optimizer will be set as default.
        Example:
        mcan_small.yml ->
            OPT: Adam
            OPT_PARAMS: {betas: '(0.9, 0.98)', eps: '1e-9'}
        """
        self.OPT = ''
        self.OPT_PARAMS = {}

    def str_to_bool(self, args):
        bool_list = ['EVAL_EVERY_EPOCH', 'TEST_SAVE_PRED', 'RESUME',
            'PIN_MEM', 'VERBOSE']
        for arg in dir(args):
            if arg in bool_list and getattr(args, arg) is not None:
                setattr(args, arg, eval(getattr(args, arg)))
        return args

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
        self.check_path(self.DATASET)
        paddle.seed(seed=self.SEED)
        if self.N_GPU < 2:
            paddle.seed(seed=self.SEED)
        else:
            paddle.seed(seed=self.SEED)
        # False = True
        np.random.seed(self.SEED)
        random.seed(self.SEED)
        if self.CKPT_PATH is not None:
            print(
                "Warning: you are now using 'CKPT_PATH' args, 'CKPT_VERSION' and 'CKPT_EPOCH' will not work"
                )
            self.CKPT_VERSION = self.CKPT_PATH.split('/')[-1] + '_' + str(
                random.randint(0, 9999999))
        self.SPLIT = self.SPLITS[self.DATASET]
        self.SPLIT['train'] = self.TRAIN_SPLIT
        if self.SPLIT['val'] in self.SPLIT['train'].split('+'
            ) or self.RUN_MODE not in ['train']:
            self.EVAL_EVERY_EPOCH = False
        if self.RUN_MODE not in ['test']:
            self.TEST_SAVE_PRED = False
        assert self.BATCH_SIZE % self.GRAD_ACCU_STEPS == 0
        self.SUB_BATCH_SIZE = int(self.BATCH_SIZE / self.GRAD_ACCU_STEPS)
        # self.EVAL_BATCH_SIZE = int(self.SUB_BATCH_SIZE / 2)
        self.EVAL_BATCH_SIZE = 1
        assert self.LOSS_FUNC in ['ce', 'bce', 'kld', 'mse']
        assert self.LOSS_REDUCTION in ['none', 'elementwise_mean', 'sum']
        self.LOSS_FUNC_NAME_DICT = {'ce': 'CrossEntropyLoss', 'bce':
            'BCEWithLogitsLoss', 'kld': 'KLDivLoss', 'mse': 'MSELoss'}
        self.LOSS_FUNC_NONLINEAR = {'ce': [None, 'flat'], 'bce': [None,
            None], 'kld': ['log_softmax', None], 'mse': [None, None]}
        self.TASK_LOSS_CHECK = {'vqa': ['bce', 'kld'], 'gqa': ['ce'],
            'clevr': ['ce']}
        assert self.LOSS_FUNC in self.TASK_LOSS_CHECK[self.DATASET
            ], self.DATASET + 'task only support' + str(self.
            TASK_LOSS_CHECK[self.DATASET]
            ) + 'loss.' + 'Modify the LOSS_FUNC in configs to get a better score.'
        assert self.OPT in ['Adam', 'Adamax', 'RMSprop', 'SGD', 'Adadelta',
            'Adagrad']
        optim = getattr(paddle.optimizer, self.OPT)
        default_params_dict = dict(zip(optim.__init__.__code__.co_varnames[
            3:optim.__init__.__code__.co_argcount], optim.__init__.
            __defaults__[1:]))

        def all(iterable):
            for element in iterable:
                if not element:
                    return False
            return True
        print(default_params_dict)
        print(self.OPT_PARAMS)
        assert all(list(map(lambda x: x in default_params_dict, self.OPT_PARAMS)))
        for key in self.OPT_PARAMS:
            if isinstance(self.OPT_PARAMS[key], str):
                self.OPT_PARAMS[key] = eval(self.OPT_PARAMS[key])
            else:
                print(
                    "To avoid ambiguity, set the value of 'OPT_PARAMS' to string type"
                    )
                exit(-1)
        self.OPT_PARAMS = {**default_params_dict, **self.OPT_PARAMS}

    def __str__(self):
        __C_str = ''
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self,
                attr), MethodType):
                __C_str += '{ %-17s }->' % attr + str(getattr(self, attr)
                    ) + '\n'
        return __C_str
