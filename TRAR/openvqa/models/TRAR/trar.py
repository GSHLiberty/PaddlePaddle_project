import sys
sys.path.append('/data2/gsh/paddlepaddle/TRAR/utils')
import paddle_aux
import paddle
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
import math
import numpy as np


class AttFlat(paddle.nn.Layer):

    def __init__(self, in_channel, glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses
        self.mlp = MLP(in_size=in_channel, mid_size=in_channel, out_size=
            glimpses, dropout_r=dropout_r, use_relu=True)
        self.linear_merge = paddle.nn.Linear(in_features=in_channel *
            glimpses, out_features=in_channel)
        self.norm = LayerNorm(in_channel)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = paddle.where(x_mask.squeeze(axis=1).squeeze(axis=1).unsqueeze
            (axis=2), att, -1000000000.0)
        att = paddle.nn.functional.softmax(x=att, axis=1)
        att_list = []
        for i in range(self.glimpses):
            att_list.append(paddle.sum(x=att[:, :, i:i + 1] * x, axis=1))
        x_atted = paddle.concat(x=att_list, axis=1)
        x_atted = self.linear_merge(x_atted)
        x_atted = self.norm(x_atted)
        return x_atted


def getImgMasks(scale=16, order=2):
    """
    :param scale: Feature Map Scale
    :param order: Local Window Size, e.g., order=2 equals to windows size (5, 5)
    :return: masks = (scale**2, scale**2)
    """
    masks = []
    _scale = scale
    assert order < _scale, 'order size be smaller than feature map scale'
    for i in range(_scale):
        for j in range(_scale):
            mask = np.ones([_scale, _scale], dtype=np.float32)
            for x in range(i - order, i + order + 1, 1):
                for y in range(j - order, j + order + 1, 1):
                    if 0 <= x < _scale and 0 <= y < _scale:
                        mask[x][y] = 0
            mask = np.reshape(mask, [_scale * _scale])
            masks.append(mask)
    masks = np.array(masks)
    masks = np.asarray(masks, dtype=np.bool_)
    return masks


def getMasks(x_mask, __C):
    mask_list = []
    ORDERS = __C.ORDERS
    
    # Ensure x_mask is of boolean type for logical operations
    x_mask = paddle.cast(x_mask, dtype=paddle.bool)

    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            # Assume getImgMasks should return a boolean mask; cast if necessary
            mask = paddle.to_tensor(getImgMasks(__C.IMG_SCALE, order), dtype=paddle.bool)

            # Perform logical OR operation
            mask = paddle.logical_or(x=x_mask, y=mask)
            mask_list.append(mask)

    return mask_list



class SoftRoutingBlock(paddle.nn.Layer):

    def __init__(self, in_channel, out_channel, pooling='attention',
        reduction=2):
        super(SoftRoutingBlock, self).__init__()
        self.pooling = pooling
        if pooling == 'attention':
            self.pool = AttFlat(in_channel)
        elif pooling == 'avg':
            self.pool = paddle.nn.AdaptiveAvgPool1D(output_size=1)
        elif pooling == 'fc':
            self.pool = paddle.nn.Linear(in_features=in_channel, out_features=1
                )
        self.mlp = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            in_channel, out_features=in_channel // reduction, bias_attr=
            False), paddle.nn.ReLU(), paddle.nn.Linear(in_features=
            in_channel // reduction, out_features=out_channel, bias_attr=True))

    def forward(self, x, tau, masks):
        if self.pooling == 'attention':
            x = self.pool(x, x_mask=self.make_mask(x))
            logits = self.mlp(x.squeeze(axis=-1))
        elif self.pooling == 'avg':
            x = x
            perm_0 = list(range(x.ndim))
            perm_0[1] = 2
            perm_0[2] = 1
            x = x.transpose(perm=perm_0)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(axis=-1))
        elif self.pooling == 'fc':
            b, _, c = x.shape
            mask = self.make_mask(x).squeeze().unsqueeze(axis=2)
            scores = self.pool(x)
            scores = paddle.where(mask, scores, -1000000000.0)
            scores = paddle.nn.functional.softmax(x=scores, axis=1)
            _x = x.mul(scores)
            x = paddle.sum(x=_x, axis=1)
            logits = self.mlp(x)
        alpha = paddle.nn.functional.softmax(x=logits, axis=-1)
        return alpha

    def make_mask(self, feature):
        return (paddle.sum(x=paddle.abs(x=feature), axis=-1) == 0).unsqueeze(
            axis=1).unsqueeze(axis=2)


class HardRoutingBlock(paddle.nn.Layer):

    def __init__(self, in_channel, out_channel, pooling='attention',
        reduction=2):
        super(HardRoutingBlock, self).__init__()
        self.pooling = pooling
        if pooling == 'attention':
            self.pool = AttFlat(in_channel)
        elif pooling == 'avg':
            self.pool = paddle.nn.AdaptiveAvgPool1D(output_size=1)
        elif pooling == 'fc':
            self.pool = paddle.nn.Linear(in_features=in_channel, out_features=1
                )
        self.mlp = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            in_channel, out_features=in_channel // reduction, bias_attr=
            False), paddle.nn.ReLU(), paddle.nn.Linear(in_features=
            in_channel // reduction, out_features=out_channel, bias_attr=True))

    def forward(self, x, tau, masks):
        if self.pooling == 'attention':
            x = self.pool(x, x_mask=self.make_mask(x))
            logits = self.mlp(x.squeeze(axis=-1))
        elif self.pooling == 'avg':
            x = x
            perm_1 = list(range(x.ndim))
            perm_1[1] = 2
            perm_1[2] = 1
            x = x.transpose(perm=perm_1)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(axis=-1))
        elif self.pooling == 'fc':
            b, _, c = x.shape
            mask = self.make_mask(x).squeeze().unsqueeze(axis=2)
            scores = self.pool(x)
            scores = paddle.where(mask, scores, -1000000000.0)
            scores = paddle.nn.functional.softmax(x=scores, axis=1)
            _x = x.mul(scores)
            x = paddle.sum(x=_x, axis=1)
            logits = self.mlp(x)
        alpha = self.gumbel_softmax(logits, -1, tau)
        return alpha

    def gumbel_softmax(self, logits, dim=-1, temperature=0.1):
        """
        Use this to replace argmax
        My input is probability distribution, multiply by 10 to get a value like logits' outputs.
        """
        gumbels = -paddle.empty_like(x=logits).exponential_().log()
        logits = (logits.log_softmax(dim=dim) + gumbels) / temperature
        return paddle.nn.functional.softmax(x=logits, axis=dim)

    def make_mask(self, feature):
        return (paddle.sum(x=paddle.abs(x=feature), axis=-1) == 0).unsqueeze(
            axis=1).unsqueeze(axis=2)


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
        x = paddle.reshape(self.linear_v(v), [n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        v = x.transpose(perm=perm_2)
        x = paddle.reshape(self.linear_k(k), [n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        k = x.transpose(perm=perm_3)
        x = paddle.reshape(self.linear_q(q), [n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_4 = list(range(x.ndim))
        perm_4[1] = 2
        perm_4[2] = 1
        q = x.transpose(perm=perm_4)
        atted = self.att(v, k, q, mask)
        x = atted
        perm_5 = list(range(x.ndim))
        perm_5[1] = 2
        perm_5[2] = 1
        atted = paddle.reshape(x.transpose(perm=perm_5), [n_batches, -1, self.__C.HIDDEN_SIZE])
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.shape[-1]
        x = key
        perm_6 = list(range(x.ndim))
        perm_6[-2] = -1
        perm_6[-1] = -2
        scores = paddle.matmul(x=query, y=x.transpose(perm=perm_6)) / math.sqrt(d_k)
        # Convert scalar to the same dtype as scores
        neg_inf = paddle.to_tensor(-1000000000.0, dtype=scores.dtype)
        if mask is not None:
            # Use paddle.where with tensors of the same dtype
            scores = paddle.where(mask, scores, neg_inf)
            # scores = paddle.where(mask, scores, -1000000000.0)
        att_map = paddle.nn.functional.softmax(x=scores, axis=-1)
        att_map = self.dropout(att_map)
        return paddle.matmul(x=att_map, y=value)


class SARoutingBlock(paddle.nn.Layer):
    """
    Self-Attention Routing Block
    """

    def __init__(self, __C):
        super(SARoutingBlock, self).__init__()
        self.__C = __C
        self.linear_v = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_k = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_q = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        self.linear_merge = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE,
            out_features=__C.HIDDEN_SIZE)
        if __C.ROUTING == 'hard':
            self.routing_block = HardRoutingBlock(__C.HIDDEN_SIZE, len(__C.
                ORDERS), __C.POOLING)
        elif __C.ROUTING == 'soft':
            self.routing_block = SoftRoutingBlock(__C.HIDDEN_SIZE, len(__C.
                ORDERS), __C.POOLING)
        self.dropout = paddle.nn.Dropout(p=__C.DROPOUT_R)

    def forward(self, v, k, q, masks, tau, training):
        n_batches = q.shape[0]
        x = v
        alphas = self.routing_block(x, tau, masks)
        if self.__C.BINARIZE:
            if not training:
                alphas = self.argmax_binarize(alphas)
        x = paddle.reshape(self.linear_v(v), [n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_7 = list(range(x.ndim))
        perm_7[1] = 2
        perm_7[2] = 1
        v = x.transpose(perm=perm_7)
        x = paddle.reshape(self.linear_k(k), [n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_8 = list(range(x.ndim))
        perm_8[1] = 2
        perm_8[2] = 1
        k = x.transpose(perm=perm_8)
        x = paddle.reshape(self.linear_q(q), [n_batches, -1, self.__C.MULTI_HEAD, int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)])
        perm_9 = list(range(x.ndim))
        perm_9[1] = 2
        perm_9[2] = 1
        q = x.transpose(perm=perm_9)
        att_list = self.routing_att(v, k, q, masks)
        att_map = paddle.einsum('bl,blcnm->bcnm', alphas, att_list)
        atted = paddle.matmul(x=att_map, y=v)
        x = atted
        perm_10 = list(range(x.ndim))
        perm_10[1] = 2
        perm_10[2] = 1
        atted = paddle.reshape(x.transpose(perm=perm_10), [n_batches, -1, self.__C.HIDDEN_SIZE])
        atted = self.linear_merge(atted)
        return atted

    def routing_att(self, value, key, query, masks):
        d_k = query.shape[-1]
        x = key
        perm_11 = list(range(x.ndim))
        perm_11[-2] = -1
        perm_11[-1] = -2
        scores = paddle.matmul(x=query, y=x.transpose(perm=perm_11)
            ) / math.sqrt(d_k)
        for i in range(len(masks)):
            mask = masks[i]

            # Convert scalar to the same dtype as scores
            neg_inf = paddle.to_tensor(-1000000000.0, dtype=scores.dtype)

            # Use paddle.where with tensors of the same dtype
            scores_temp = paddle.where(mask, scores, neg_inf)

            # scores_temp = paddle.where(mask, scores, -1000000000.0)
            att_map = paddle.nn.functional.softmax(x=scores_temp, axis=-1)
            att_map = self.dropout(att_map)
            if i == 0:
                att_list = att_map.unsqueeze(axis=1)
            else:
                att_list = paddle.concat(x=(att_list, att_map.unsqueeze(
                    axis=1)), axis=1)
        return att_list

    def argmax_binarize(self, alphas):
        n = alphas.shape[0]
        out = paddle.zeros_like(x=alphas)
        indexes = alphas.argmax(axis=-1)
        out[paddle.arange(end=n), indexes] = 1
        return out


class FFN(paddle.nn.Layer):

    def __init__(self, __C):
        super(FFN, self).__init__()
        self.mlp = MLP(in_size=__C.HIDDEN_SIZE, mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE, dropout_r=__C.DROPOUT_R, use_relu=True)

    def forward(self, x):
        return self.mlp(x)


class Encoder(paddle.nn.Layer):

    def __init__(self, __C):
        super(Encoder, self).__init__()
        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)
        self.dropout1 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(self.mhatt(y, y, y, y_mask)))
        y = self.norm2(y + self.dropout2(self.ffn(y)))
        return y


class TRAR(paddle.nn.Layer):

    def __init__(self, __C):
        super(TRAR, self).__init__()
        self.mhatt1 = SARoutingBlock(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)
        self.dropout1 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout3 = paddle.nn.Dropout(p=__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_masks, y_mask, tau, training):
        x = self.norm1(x + self.dropout1(self.mhatt1(v=x, k=x, q=x, masks=
            x_masks, tau=tau, training=training)))
        x = self.norm2(x + self.dropout2(self.mhatt2(v=y, k=y, q=x, mask=
            y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class TRAR_ED(paddle.nn.Layer):

    def __init__(self, __C):
        super(TRAR_ED, self).__init__()
        self.__C = __C
        self.tau = __C.TAU_MAX
        self.training = True
        self.enc_list = paddle.nn.LayerList(sublayers=[Encoder(__C) for _ in
            range(__C.LAYER)])
        self.dec_list = paddle.nn.LayerList(sublayers=[TRAR(__C) for _ in
            range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        x_masks = getMasks(x_mask, self.__C)
        for enc in self.enc_list:
            y = enc(y, y_mask)
        for dec in self.dec_list:
            x = dec(x, y, x_masks, y_mask, self.tau, self.training)
        return y, x

    def set_tau(self, tau):
        self.tau = tau

    def set_training_status(self, training):
        self.training = training
