import paddle
import math
import copy


def clones(module, N):
    """Produce N identical layers."""
    return paddle.nn.LayerList(sublayers=[copy.deepcopy(module) for _ in
        range(N)])


class ScaledDotProductAttention(paddle.nn.Layer):
    """
    Compute 'Scaled Dot Product Attention'
    """

    def __init__(self, dropout=0.0):
        """
        :param dropout: attention dropout rate
        """
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        d_k = query.shape[-1]
        x = key
        perm_18 = list(range(x.ndim))
        perm_18[-2] = -1
        perm_18[-1] = -2
        scores = paddle.matmul(x=query, y=x.transpose(perm=perm_18)
            ) / math.sqrt(d_k)
        if mask is not None:
            scores = paddle.where(mask == 0, paddle.to_tensor([-1000000000.0], dtype='float32'), scores)
        p_attn = paddle.nn.functional.softmax(x=scores, axis=-1)
        p_attn = paddle.nn.functional.dropout(x=p_attn, p=self.dropout)
        return paddle.matmul(x=p_attn, y=value), p_attn


class MultiHeadAttention(paddle.nn.Layer):
    """
    Compute 'Multi-Head Attention'
    When we calculate attentions, usually key and value are the same tensor.
    For self-attention, query, key, value are all the same tensor.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of heads
        :param d_model: hidden size
        :param dropout: attention dropout rate
        """
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(paddle.nn.Linear(in_features=d_model,
            out_features=d_model), 4)
        self.attention = ScaledDotProductAttention(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(axis=1)
        nbatches = query.shape[0]

        processed_tensors = [paddle.transpose(l(x).reshape([nbatches, -1, self.h, self.d_k]), perm=[0, 2, 1, 3]) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = paddle.stack(processed_tensors)
        x, self.attn = self.attention(query, key, value, mask=mask)
        # 使用 paddle.transpose 函数将张量的维度调整为 (nbatches, self.h, -1, self.d_k)
        x = paddle.transpose(x, perm=[0, 2, 1, 3])

        # 使用 paddle.contiguous 函数来获取连续内存的张量，以保证后续计算的正确性
        x = paddle.contiguous(x)

        # 使用 paddle.reshape 函数将张量的形状调整为 (nbatches, -1, self.h * self.d_k)
        x = paddle.reshape(x, shape=[nbatches, -1, self.h * self.d_k])

        # 使用 self.linears[-1] 对处理后的张量 x 进行线性变换
        output = self.linears[-1](x)

        # 返回处理后的张量 output 和 self.attn
        return output, self.attn


if __name__ == '__main__':
    n_head = 8
    d_model = 128
    d_k = d_model // n_head
    d_v = d_model // n_head
    batch_num = 10
    len_q = 20
    len_k = 30
    q = paddle.rand(shape=[batch_num, len_q, d_model])
    k = paddle.rand(shape=[batch_num, len_k, d_model])
    v = k
    model = MultiHeadAttention(n_head, d_model, dropout=0.1)
    output, attn = model(q, k, v)
    print(output.shape)
    print(attn.shape)
