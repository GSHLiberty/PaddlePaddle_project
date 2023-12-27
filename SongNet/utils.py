import paddle
from collections import defaultdict
import math


def gelu(x):
    cdf = 0.5 * (1.0 + paddle.erf(x=x / math.sqrt(2.0)))
    return cdf * x


class LayerNorm(paddle.nn.Layer):

    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()

        out_2 = paddle.create_parameter(shape=paddle.to_tensor(data=hidden_size).shape, 
                                        # dtype=paddle.to_tensor(data=hidden_size).numpy().dtype, 
                                        dtype = 'float32',
                                        default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(data=hidden_size)))
        out_2.stop_gradient = not True
        self.weight = out_2
        out_3 = paddle.create_parameter(shape=paddle.to_tensor(data=hidden_size).shape, 
                                        # dtype=paddle.to_tensor(data=hidden_size).numpy().dtype, 
                                        dtype = 'float32',
                                        default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(data=hidden_size)))
        out_3.stop_gradient = not True
        self.bias = out_3
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = paddle.nn.initializer.Constant(value=1.0)
        init_Constant(self.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.bias)

    def forward(self, x):
        u = x.mean(axis=-1, keepdim=True)
        s = (x - u).pow(y=2).mean(axis=-1, keepdim=True)
        x = (x - u) / paddle.sqrt(x=s + self.eps)
        return self.weight * x + self.bias


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda : 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    if not hasattr(module_instance, '_guyu_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._guyu_instance_id = INCREMENTAL_STATE_INSTANCE_ID[
            module_name]
    return '{}.{}.{}'.format(module_name, module_instance._guyu_instance_id,
        key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value
