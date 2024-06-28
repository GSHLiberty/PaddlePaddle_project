import paddle


class WarmupOptimizer(object):

    def __init__(self, lr_base, optimizer, data_size, batch_size, warmup_epoch
        ):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        self.warmup_epoch = warmup_epoch

    def step(self):
        self._step += 1
        rate = self.rate()

        self.optimizer.set_lr(rate)  # 设置新的学习率

        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.clear_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        if step <= int(self.data_size / self.batch_size * (self.
            warmup_epoch + 1) * 0.25):
            r = self.lr_base * 1 / (self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.
            warmup_epoch + 1) * 0.5):
            r = self.lr_base * 2 / (self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.
            warmup_epoch + 1) * 0.75):
            r = self.lr_base * 3 / (self.warmup_epoch + 1)
        else:
            r = self.lr_base
        return r

def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.LR_BASE

    # 确保优化器名称有效
    try:
        std_optim_class = getattr(paddle.optimizer, __C.OPT)
    except AttributeError:
        raise ValueError(f"Optimizer '{__C.OPT}' is not recognized by PaddlePaddle.")

    # 过滤需要更新的模型参数
    params = [p for p in model.parameters() if not p.stop_gradient]
    # 构建优化器参数
    optim_args = {'parameters': params, 'learning_rate': lr_base}
    if 'parameters' in __C.OPT_PARAMS:
        del __C.OPT_PARAMS['parameters']
    optim_args.update(__C.OPT_PARAMS)

    # 创建优化器实例
    optimizer = std_optim_class(**optim_args)

    # 创建带预热的优化器
    optim = WarmupOptimizer(
        lr_base,
        optimizer,
        data_size,
        __C.BATCH_SIZE,
        __C.WARMUP_EPOCH
    )

    return optim


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r
