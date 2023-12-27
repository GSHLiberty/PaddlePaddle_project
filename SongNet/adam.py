import sys
sys.path.append('/data2/gsh/paddlepaddle/SongNet/utils')
from utils import paddle_aux
import paddle


class AdamWeightDecayOptimizer(paddle.optimizer.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.
    https://github.com/google-research/bert/blob/master/optimization.py
    https://raw.githubusercontent.com/pytorch/pytorch/v1.0.0/torch/optim/adam.py"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
        weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format
                (betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format
                (betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=
            weight_decay, amsgrad=amsgrad)
        super(AdamWeightDecayOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWeightDecayOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse():
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                        )
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = paddle.zeros_like(x=p.data)
                    state['exp_avg_sq'] = paddle.zeros_like(x=p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = paddle.zeros_like(x=p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.multiply_(y=paddle.to_tensor(beta1)).add_(1 - beta1, grad)
                exp_avg_sq.multiply_(y=paddle.to_tensor(beta2))
                exp_avg_sq = exp_avg_sq + (1 - beta2) * grad * grad
                if amsgrad:
                    paddle_aux.max(max_exp_avg_sq, exp_avg_sq, out=
                        max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(y=paddle.to_tensor(
                        group['eps']))
                else:
                    denom = exp_avg_sq.sqrt().add_(y=paddle.to_tensor(group
                        ['eps']))
                update = (exp_avg / denom).add_(group['weight_decay'], p.data)
                p.data.add_(-group['lr'], update)
        return loss
