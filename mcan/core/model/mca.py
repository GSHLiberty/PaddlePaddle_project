import paddle
from core.model.net_utils import FC, MLP, LayerNorm
import math


class MHAtt(paddle.nn.Layer):

    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C
        self.linear_v = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_k = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_q = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_merge = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.dropout = paddle.nn.Dropout(p=__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        # import ipdb
        # ipdb.set_trace()
        n_batches = q.shape[0]
        x = paddle.reshape(self.linear_v(v), shape=[n_batches, -1, self.__C.MULTI_HEAD, self.__C.HIDDEN_SIZE_HEAD])
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        v = x.transpose(perm=perm_0)
        x = paddle.reshape(self.linear_k(k), shape=[n_batches, -1, self.__C.MULTI_HEAD, self.__C.HIDDEN_SIZE_HEAD])
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        k = x.transpose(perm=perm_1)
        x = paddle.reshape(self.linear_q(q), shape=[n_batches, -1, self.__C.MULTI_HEAD, self.__C.HIDDEN_SIZE_HEAD])
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        q = x.transpose(perm=perm_2)
        atted = self.att(v, k, q, mask)
        x = atted
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        atted = paddle.reshape(x.transpose(perm=perm_3), shape=[n_batches, -1, self.__C.HIDDEN_SIZE])
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.shape[-1]
        scores = paddle.matmul(query, paddle.transpose(key, perm=[0, 1, 3, 2])) / paddle.sqrt(paddle.to_tensor(float(d_k)))
        if mask is not None:
            scores = paddle.where(mask, paddle.full_like(scores, -1e9), scores)
        att_map = paddle.nn.functional.softmax(x=scores, axis=-1)
        att_map = self.dropout(att_map)
        return paddle.matmul(x=att_map, y=value)


class FFN(paddle.nn.Layer):

    def __init__(self, __C):
        super(FFN, self).__init__()
        self.mlp = MLP(in_size=__C.HIDDEN_SIZE, mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE, dropout_r=__C.DROPOUT_R, use_relu=True)

    def forward(self, x):
        return self.mlp(x)


class SA(paddle.nn.Layer):

    def __init__(self, __C):
        super(SA, self).__init__()
        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)
        self.dropout1 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class SGA(paddle.nn.Layer):

    def __init__(self, __C):
        super(SGA, self).__init__()
        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)
        self.dropout1 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout3 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class MCA_ED(paddle.nn.Layer):

    def __init__(self, __C):
        super(MCA_ED, self).__init__()
        self.enc_list = paddle.nn.LayerList(sublayers=[SA(__C) for _ in
            range(__C.LAYER)])
        self.dec_list = paddle.nn.LayerList(sublayers=[SGA(__C) for _ in
            range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)
        return x, y
