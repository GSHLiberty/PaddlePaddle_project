import paddle


class FC(paddle.nn.Layer):

    def __init__(self, in_size, out_size, dropout_r=0.0, use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = paddle.nn.Linear(in_features=in_size, out_features=
            out_size)
        if use_relu:
            self.relu = paddle.nn.ReLU()
        if dropout_r > 0:
            self.dropout = paddle.nn.Dropout(p=dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x


class MLP(paddle.nn.Layer):

    def __init__(self, in_size, mid_size, out_size, dropout_r=0.0, use_relu
        =True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = paddle.nn.Linear(in_features=mid_size, out_features=
            out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(paddle.nn.Layer):

    def __init__(self, size, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.eps = eps
        x = paddle.ones(shape=(size,))
        out_0 = paddle.create_parameter(shape=x.shape, dtype=str(x.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(x))
        out_0.stop_gradient = not True
        self.a_2 = out_0
        y = paddle.zeros(shape=(size,))
        out_1 = paddle.create_parameter(shape=y.shape, dtype=str(y.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(y))
        out_1.stop_gradient = not True
        self.b_2 = out_1

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        std = x.std(axis=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
