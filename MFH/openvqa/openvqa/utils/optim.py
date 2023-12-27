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

    # def step(self):
    #     self._step += 1
    #     rate = self.rate()
    #     for p in self.optimizer.param_groups:
    #         p['lr'] = rate
    #     self._rate = rate
    #     self.optimizer.step()
    def step(self):
        self._step += 1
        rate = self.rate()

        self.optimizer.set_lr(rate)  # 设置新的学习率

        self._rate = rate
        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.clear_grad()
#         """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        self.optimizer.zero_grad()

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


# def get_optim(__C, model, data_size, lr_base=None):
#     if lr_base is None:
#         lr_base = __C.LR_BASE
#     std_optim = getattr(paddle.optimizer, __C.OPT)
#     params = filter(lambda p: not p.stop_gradient, model.parameters())
#     eval_str = 'params'
#     for key in __C.OPT_PARAMS:
#         eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
#     optim = WarmupOptimizer(lr_base, eval('std_optim' + '(' + eval_str +
#         ')'), data_size, __C.BATCH_SIZE, __C.WARMUP_EPOCH)
#     return optim
# import paddle

def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.LR_BASE

    # 获取优化器
    std_optim_class = getattr(paddle.optimizer, __C.OPT)

    # 在PaddlePaddle中，不需要梯度的参数有stop_gradient属性设置为True
    params = [p for p in model.parameters() if not p.stop_gradient]
    # print(params)
    # 使用字典来构建优化器参数
    optim_args = {'parameters': params, 'learning_rate': 0.0}
    if 'parameters' in __C.OPT_PARAMS:
        del __C.OPT_PARAMS['parameters']
    optim_args.update(__C.OPT_PARAMS)
    # 使用**进行字典拆包来创建优化器
    optimizer = std_optim_class(**optim_args)

    # 注意：这里假设WarmupOptimizer的__init__方法接受优化器作为其中一个参数
    # 如果不是这样，你需要相应地调整代码
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
