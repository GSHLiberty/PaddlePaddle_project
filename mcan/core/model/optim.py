import paddle


class WarmupOptimizer(object):

    def __init__(self, lr_base, optimizer, data_size, batch_size):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.clear_grad()
#         """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        if step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 1 / 4.0
        elif step <= int(self.data_size / self.batch_size * 2):
            r = self.lr_base * 2 / 4.0
        elif step <= int(self.data_size / self.batch_size * 3):
            r = self.lr_base * 3 / 4.0
        else:
            r = self.lr_base
        return r


def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.LR_BASE
    return WarmupOptimizer(lr_base, paddle.optimizer.Adam(parameters=filter
        (lambda p: not p.stop_gradient, model.parameters()), learning_rate=
        0.0, epsilon=__C.OPT_EPS, beta1=__C.OPT_BETAS[0], beta2=__C.OPT_BETAS
        [1], weight_decay=0.0), data_size, __C.BATCH_SIZE)


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r
