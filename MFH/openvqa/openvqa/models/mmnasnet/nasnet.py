import paddle
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
import math


class RelMHAtt(paddle.nn.Layer):

    def __init__(self, __C):
        super(RelMHAtt, self).__init__()
        self.__C = __C
        self.HBASE = __C.REL_HBASE
        self.HHEAD = int(__C.HIDDEN_SIZE / __C.REL_HBASE)
        self.linear_v = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_k = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_q = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_merge = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_r = paddle.nn.Linear(in_features=__C.REL_SIZE,
            out_features=self.HHEAD, bias_attr=True)
        self.dropout = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.relu = paddle.nn.ReLU()

    def forward(self, v, k, q, mask=None, rel_embed=None):
        assert rel_embed is not None
        n_batches = q.shape[0]
        x = paddle.reshape(self.linear_v(v), shape=[n_batches, -1, self.HHEAD, self.HBASE])
        perm_9 = list(range(x.ndim))
        perm_9[1] = 2
        perm_9[2] = 1
        v = x.transpose(perm=perm_9)
        x = paddle.reshape(self.linear_k(k), shape=[n_batches, -1, self.HHEAD, self.HBASE])
        perm_10 = list(range(x.ndim))
        perm_10[1] = 2
        perm_10[2] = 1
        k = x.transpose(perm=perm_10)
        x = paddle.reshape(self.linear_q(q), shape=[n_batches, -1, self.HHEAD, self.HBASE])
        perm_11 = list(range(x.ndim))
        perm_11[1] = 2
        perm_11[2] = 1
        q = x.transpose(perm=perm_11)
        r = self.relu(self.linear_r(rel_embed)).transpose(perm=[0, 3, 1, 2])
        d_k = q.shape[-1]
        x = k
        perm_12 = list(range(x.ndim))
        perm_12[-2] = -1
        perm_12[-1] = -2
        scores = paddle.matmul(x=q, y=x.transpose(perm=perm_12)) / math.sqrt(
            d_k)
        scores = paddle.log(x=paddle.clip(x=r, min=1e-06)) + scores
        if mask is not None:
            scores = paddle.where(mask, paddle.full_like(scores, -1e9), scores)
        att_map = paddle.nn.functional.softmax(x=scores, axis=-1)
        att_map = self.dropout(att_map)
        atted = paddle.matmul(x=att_map, y=v)
        x = atted
        perm_13 = list(range(x.ndim))
        perm_13[1] = 2
        perm_13[2] = 1
        atted = paddle.reshape(x.transpose(perm=perm_13), shape=[n_batches, -1, self.__C.HIDDEN_SIZE])
        atted = self.linear_merge(atted)
        return atted


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
        n_batches = q.shape[0]
        x = paddle.reshape(self.linear_v(v), shape=[n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_14 = list(range(x.ndim))
        perm_14[1] = 2
        perm_14[2] = 1
        v = x.transpose(perm=perm_14)
        x = paddle.reshape(self.linear_k(k), shape=[n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_15 = list(range(x.ndim))
        perm_15[1] = 2
        perm_15[2] = 1
        k = x.transpose(perm=perm_15)
        x = paddle.reshape(self.linear_q(q), shape=[n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_16 = list(range(x.ndim))
        perm_16[1] = 2
        perm_16[2] = 1
        q = x.transpose(perm=perm_16)
        atted = self.att(v, k, q, mask)
        x = atted
        perm_17 = list(range(x.ndim))
        perm_17[1] = 2
        perm_17[2] = 1
        atted = paddle.reshape(x.transpose(perm=perm_17), shape=[n_batches, -1, self.__C.HIDDEN_SIZE])
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.shape[-1]
        x = key
        perm_18 = list(range(x.ndim))
        perm_18[-2] = -1
        perm_18[-1] = -2
        scores = paddle.matmul(x=query, y=x.transpose(perm=perm_18)
            ) / math.sqrt(d_k)
        if mask is not None:
            scores = paddle.where(mask, paddle.full_like(scores, -1e9), scores)
        att_map = paddle.nn.functional.softmax(x=scores, axis=-1)
        att_map = self.dropout(att_map)
        return paddle.matmul(x=att_map, y=value)


class FFN(paddle.nn.Layer):

    def __init__(self, __C):
        super(FFN, self).__init__()
        self.mlp = MLP(in_size=__C.HIDDEN_SIZE, mid_size=__C.HIDDEN_SIZE * 
            4, out_size=__C.HIDDEN_SIZE, dropout_r=__C.DROPOUT_R, use_relu=True
            )
        self.dropout = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, arg1, arg2, arg3, arg4):
        x = self.norm(x + self.dropout(self.mlp(x)))
        return x


class SA(paddle.nn.Layer):

    def __init__(self, __C, size=1024):
        super(SA, self).__init__()
        self.mhatt = MHAtt(__C)
        self.dropout = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, arg1, y_mask, arg2, arg3):
        y = self.norm(y + self.dropout(self.mhatt(y, y, y, y_mask)))
        return y


class RSA(paddle.nn.Layer):

    def __init__(self, __C, size=1024):
        super(RSA, self).__init__()
        self.mhatt = RelMHAtt(__C)
        self.dropout = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, arg1, x_mask, arg2, rela):
        x = self.norm(x + self.dropout(self.mhatt(x, x, x, x_mask, rela)))
        return x


class GA(paddle.nn.Layer):

    def __init__(self, __C):
        super(GA, self).__init__()
        self.mhatt = MHAtt(__C)
        self.dropout = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask, rela):
        x = self.norm(x + self.dropout(self.mhatt(v=y, k=y, q=x, mask=y_mask)))
        return x


class NAS_ED(paddle.nn.Layer):

    def __init__(self, __C):
        super(NAS_ED, self).__init__()
        enc = __C.ARCH['enc']
        dec = __C.ARCH['dec']
        self.enc_list = paddle.nn.LayerList(sublayers=[eval(layer)(__C) for
            layer in enc])
        self.dec_list = paddle.nn.LayerList(sublayers=[eval(layer)(__C) for
            layer in dec])

    def forward(self, y, x, y_mask, x_mask, rela):
        for enc in self.enc_list:
            y = enc(y, None, y_mask, None, None)
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask, rela)
        return y, x
