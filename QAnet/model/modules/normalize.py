import paddle
"""
Layer normalization.
"""


class LayerNormalization(paddle.nn.Layer):
    """Construct a layernorm module."""

    def __init__(self, features, eps=1e-06):
        super().__init__()
        out_5 = paddle.create_parameter(shape=paddle.ones(shape=features).
            shape, dtype=paddle.ones(shape=features).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(
            shape=features)))
        out_5.stop_gradient = not True
        self.a_2 = out_5
        out_6 = paddle.create_parameter(shape=paddle.zeros(shape=features).
            shape, dtype=paddle.zeros(shape=features).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(
            shape=features)))
        out_6.stop_gradient = not True
        self.b_2 = out_6
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        std = x.std(axis=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
