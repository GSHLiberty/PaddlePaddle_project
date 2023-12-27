import paddle


class LayerNorm(paddle.nn.Layer):

    def __init__(self, size, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.eps = eps
        out_2 = paddle.create_parameter(shape=paddle.ones(shape=size).shape,
            dtype=paddle.ones(shape=size).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(
            shape=size)))
        out_2.stop_gradient = not True
        self.a_2 = out_2
        out_3 = paddle.create_parameter(shape=paddle.zeros(shape=size).
            shape, dtype=paddle.zeros(shape=size).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(
            shape=size)))
        out_3.stop_gradient = not True
        self.b_2 = out_3

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        std = x.std(axis=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
