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
