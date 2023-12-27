import paddle
import math


class PositionalEncoding(paddle.nn.Layer):
    """
    Add position information to input tensor.
    :Examples:
        >>> m = PositionalEncoding(d_model=6, max_len=10, dropout=0)
        >>> input = torch.randn(3, 10, 6)
        >>> output = m(input)
    """

    def __init__(self, d_model, dropout=0, max_len=5000):
        """
        :param d_model: same with input hidden size
        :param dropout: dropout rate
        :param max_len: maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = paddle.nn.Dropout(p=dropout)
        pe = paddle.zeros(shape=[max_len, d_model])
        position = paddle.arange(start=0, end=max_len).unsqueeze(axis=1)
        div_term = paddle.exp(x=paddle.arange(start=0, end=d_model, step=2) *
            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(x=position * div_term)
        pe[:, 1::2] = paddle.cos(x=position * div_term)
        pe = pe.unsqueeze(axis=0)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x):
        """
        :Input: (batch_num, seq_length, hidden_size)
        :Output: (batch_num, seq_length, hidden_size)
        """
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)
