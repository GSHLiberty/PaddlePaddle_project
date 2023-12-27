import sys
sys.path.append('/home/gsh/paddle_project/QAnet/utils')
from utils import paddle_aux
import paddle
import paddle.nn
"""
Main model architecture.
reference: https://github.com/andy840314/QANet-pytorch-
"""
import math
import numpy as np
from .modules.cnn import DepthwiseSeparableConv
from .modules.attention import MultiHeadAttention
from .modules.position import PositionalEncoding
device = str('cuda:2' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')


def mask_logits(target, mask):
    mask = mask.astype('float32')
    return target * mask + (1 - mask) * -1e+30


class Initialized_Conv1d(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, groups=1, relu=False, bias=False):
        super().__init__()
        self.out = paddle.nn.Conv1D(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, groups=groups, bias_attr=bias)
        self.relu = relu
        if relu is True:
            self.relu = True
            paddle.nn.initializer.KaimingNormal(self.out.weight, nonlinearity='relu')
        else:
            # self.relu = False
            # def _calculate_fan_in_and_fan_out(tensor):
            #     dimensions = tensor.dim()
            #     if dimensions < 2:
            #         raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

            #     num_input_fmaps = tensor.size(1)
            #     num_output_fmaps = tensor.size(0)
            #     receptive_field_size = 1
            #     if tensor.dim() > 2:
            #         receptive_field_size = tensor[0][0].numel()
            #     fan_in = num_input_fmaps * receptive_field_size
            #     fan_out = num_output_fmaps * receptive_field_size

            #     return fan_in, fan_out
            
            # fan_in, fan_out = _calculate_fan_in_and_fan_out(self.out.weight)
            # gain=1.
            # std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            # with paddle.no_grad():
            #     self.out.weight.set_value(paddle.normal(mean=0.0, std=std, shape=self.out.weight.shape))

            paddle.nn.initializer.XavierUniform(self.out.weight) ###########

    def forward(self, x):
        if self.relu is True:
            return paddle.nn.functional.relu(x=self.out(x))
        else:
            return self.out(x)


def PosEncoder(x, min_timescale=1.0, max_timescale=10000.0):
    x = x
    perm_0 = list(range(x.ndim))
    perm_0[1] = 2
    perm_0[2] = 1
    x = x.transpose(perm=perm_0)
    length = x.shape[1]
    channels = x.shape[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    x = x + signal.to(x.place.gpu_device_id())
    perm_1 = list(range(x.ndim))
    perm_1[1] = 2
    perm_1[2] = 1
    return x.transpose(perm=perm_1)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=
    10000.0):
    position = paddle.arange(end=length).astype('float32')
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(
        min_timescale)) / (float(num_timescales) - 1)
    inv_timescales = min_timescale * paddle.exp(x=paddle.arange(end=
        num_timescales).astype('float32') * -log_timescale_increment)
    scaled_time = position.unsqueeze(axis=1) * inv_timescales.unsqueeze(axis=0)
    signal = paddle.concat(x=[paddle.sin(x=scaled_time), paddle.cos(x=
        scaled_time)], axis=1)
    m = paddle.nn.ZeroPad2D(padding=(0, channels % 2, 0, 0))
    signal = m(signal)
    """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    signal = paddle.reshape(signal, [1, length, channels])
    return signal


class DepthwiseSeparableConv(paddle.nn.Layer):

    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = paddle.nn.Conv1D(in_channels=in_ch,
            out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2,
            bias_attr=False)
        self.pointwise_conv = paddle.nn.Conv1D(in_channels=in_ch,
            out_channels=out_ch, kernel_size=1, padding=0, bias_attr=bias)

    def forward(self, x):
        return paddle.nn.functional.relu(x=self.pointwise_conv(self.
            depthwise_conv(x)))


class Highway(paddle.nn.Layer):

    def __init__(self, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = paddle.nn.LayerList(sublayers=[Initialized_Conv1d(
            size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = paddle.nn.LayerList(sublayers=[Initialized_Conv1d(size,
            size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        dropout = 0.1
        for i in range(self.n):
            gate = paddle.nn.functional.sigmoid(x=self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = paddle.nn.functional.dropout(x=nonlinear, p=dropout,
                training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x


class SelfAttention(paddle.nn.Layer):

    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(in_channels=d_model,
            out_channels=d_model * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model,
            out_channels=d_model, kernel_size=1, relu=False, bias=False)
        bias = paddle.empty(shape=[1])
        constant_init = paddle.nn.initializer.Constant(value=0.0)
        constant_init(bias)
        out_0 = paddle.create_parameter(shape=bias.shape, dtype=bias.numpy(
            ).dtype, default_initializer=paddle.nn.initializer.Assign(bias))
        out_0.stop_gradient = not True
        self.bias = out_0

    def forward(self, queries, mask):
        memory = queries
        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        x = memory
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        memory = x.transpose(perm=perm_2)
        x = query
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        query = x.transpose(perm=perm_3)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in
            paddle.split(x=memory, num_or_sections=memory.shape[2] // self.
            d_model, axis=2)]
        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        x = self.combine_last_two_dim(x.transpose(perm=[0, 2, 1, 3]))
        perm_4 = list(range(x.ndim))
        perm_4[1] = 2
        perm_4[2] = 1
        return x.transpose(perm=perm_4)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = paddle.matmul(x=q, y=k.transpose(perm=[0, 1, 3, 2]))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [(x if x != None else -1) for x in list(logits.shape)]
            """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            mask = paddle.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
            logits = mask_logits(logits, mask)
        weights = paddle.nn.functional.softmax(x=logits, axis=-1)
        weights = paddle.nn.functional.dropout(x=weights, p=self.dropout, training=self.training)
        return paddle.matmul(x=weights, y=v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.shape)
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        ret = paddle.reshape(x, new_shape)
        return ret.transpose(perm=[0, 2, 1, 3])

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.shape)
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        ret = paddle.reshape(x, new_shape)
        return ret


class Embedding(paddle.nn.Layer):

    def __init__(self, wemb_dim, cemb_dim, d_model, dropout_w=0.1,
        dropout_c=0.05):
        super().__init__()
        self.conv2d = paddle.nn.Conv2D(in_channels=cemb_dim, out_channels=
            d_model, kernel_size=(1, 5), padding=0, bias_attr=True)
        paddle.nn.initializer.KaimingNormal(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(wemb_dim + d_model, d_model, bias=
            False)
        self.high = Highway(2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.transpose(perm=[0, 3, 1, 2])
        ch_emb = paddle.nn.functional.dropout(x=ch_emb, p=self.dropout_c,
            training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = paddle.nn.functional.relu(x=ch_emb)
        ch_emb, _ = paddle.max(x=ch_emb, axis=3), paddle.argmax(x=ch_emb,
            axis=3)
        wd_emb = paddle.nn.functional.dropout(x=wd_emb, p=self.dropout_w,
            training=self.training)
        x = wd_emb
        perm_5 = list(range(x.ndim))
        perm_5[1] = 2
        perm_5[2] = 1
        wd_emb = x.transpose(perm=perm_5)
        emb = paddle.concat(x=[ch_emb, wd_emb], axis=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(paddle.nn.Layer):

    def __init__(self, conv_num, d_model, num_head, k, dropout=0.1):
        super().__init__()
        self.convs = paddle.nn.LayerList(sublayers=[DepthwiseSeparableConv(
            d_model, d_model, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = paddle.nn.LayerList(sublayers=[paddle.nn.LayerNorm(
            normalized_shape=d_model) for _ in range(conv_num)])
        self.norm_1 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.norm_2 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        dropout = self.dropout
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            x = out
            perm_6 = list(range(x.ndim))
            perm_6[1] = 2
            perm_6[2] = 1
            x = self.norm_C[i](x.transpose(perm=perm_6))
            perm_7 = list(range(x.ndim))
            perm_7[1] = 2
            perm_7[2] = 1
            out = x.transpose(perm=perm_7)
            if i % 2 == 0:
                out = paddle.nn.functional.dropout(x=out, p=dropout,
                    training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout * float(l) /
                total_layers)
            l += 1
        res = out
        x = out
        perm_8 = list(range(x.ndim))
        perm_8[1] = 2
        perm_8[2] = 1
        x = self.norm_1(x.transpose(perm=perm_8))
        perm_9 = list(range(x.ndim))
        perm_9[1] = 2
        perm_9[2] = 1
        out = x.transpose(perm=perm_9)
        out = paddle.nn.functional.dropout(x=out, p=dropout, training=self.
            training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        l += 1
        res = out
        x = out
        perm_10 = list(range(x.ndim))
        perm_10[1] = 2
        perm_10[2] = 1
        x = self.norm_2(x.transpose(perm=perm_10))
        perm_11 = list(range(x.ndim))
        perm_11[1] = 2
        perm_11[2] = 1
        out = x.transpose(perm=perm_11)
        out = paddle.nn.functional.dropout(x=out, p=dropout, training=self.
            training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = paddle.empty(shape=[1]).uniform_(min=0, max=1) < dropout
            if pred:
                return residual
            else:
                return paddle.nn.functional.dropout(x=inputs, p=dropout,
                    training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(paddle.nn.Layer):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = paddle.empty(shape=[d_model, 1])
        w4Q = paddle.empty(shape=[d_model, 1])
        w4mlu = paddle.empty(shape=[1, 1, d_model])
        paddle.nn.initializer.XavierUniform(w4C)
        paddle.nn.initializer.XavierUniform(w4Q)
        paddle.nn.initializer.XavierUniform(w4mlu)
        out_1 = paddle.create_parameter(shape=w4C.shape, dtype=w4C.numpy().
            dtype, default_initializer=paddle.nn.initializer.Assign(w4C))
        out_1.stop_gradient = not True
        self.w4C = out_1
        out_2 = paddle.create_parameter(shape=w4Q.shape, dtype=w4Q.numpy().
            dtype, default_initializer=paddle.nn.initializer.Assign(w4Q))
        out_2.stop_gradient = not True
        self.w4Q = out_2
        out_3 = paddle.create_parameter(shape=w4mlu.shape, dtype=w4mlu.
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (w4mlu))
        out_3.stop_gradient = not True
        self.w4mlu = out_3
        bias = paddle.empty(shape=[1])
        constant_init = paddle.nn.initializer.Constant(value=0.0)
        constant_init(bias)
        out_4 = paddle.create_parameter(shape=bias.shape, dtype=bias.numpy(
            ).dtype, default_initializer=paddle.nn.initializer.Assign(bias))
        out_4.stop_gradient = not True
        self.bias = out_4
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        x = C
        perm_12 = list(range(x.ndim))
        perm_12[1] = 2
        perm_12[2] = 1
        C = x.transpose(perm=perm_12)
        x = Q
        perm_13 = list(range(x.ndim))
        perm_13[1] = 2
        perm_13[2] = 1
        Q = x.transpose(perm=perm_13)
        batch_size_c = C.shape[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        Cmask = paddle.reshape(Cmask, [batch_size_c, Lc, 1])
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        Qmask = paddle.reshape(Qmask, [batch_size_c, 1, Lq])
        S1 = paddle.nn.functional.softmax(x=mask_logits(S, Qmask), axis=2)
        S2 = paddle.nn.functional.softmax(x=mask_logits(S, Cmask), axis=1)
        A = paddle.bmm(x=S1, y=Q)
        x = S2
        perm_14 = list(range(x.ndim))
        perm_14[1] = 2
        perm_14[2] = 1
        B = paddle.bmm(x=paddle.bmm(x=S1, y=x.transpose(perm=perm_14)), y=C)
        out = paddle.concat(x=[C, A, C * A, C * B], axis=2)
        x = out
        perm_15 = list(range(x.ndim))
        perm_15[1] = 2
        perm_15[2] = 1
        return x.transpose(perm=perm_15)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = paddle.nn.functional.dropout(x=C, p=dropout, training=self.training
            )
        Q = paddle.nn.functional.dropout(x=Q, p=dropout, training=self.training
            )
        subres0 = paddle.matmul(x=C, y=self.w4C).expand(shape=[-1, -1, Lq])
        x = paddle.matmul(x=Q, y=self.w4Q)
        perm_16 = list(range(x.ndim))
        perm_16[1] = 2
        perm_16[2] = 1
        subres1 = x.transpose(perm=perm_16).expand(shape=[-1, Lc, -1])
        x = Q
        perm_17 = list(range(x.ndim))
        perm_17[1] = 2
        perm_17[2] = 1
        subres2 = paddle.matmul(x=C * self.w4mlu, y=x.transpose(perm=perm_17))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(paddle.nn.Layer):

    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model * 2, 1)
        self.w2 = Initialized_Conv1d(d_model * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = paddle.concat(x=[M1, M2], axis=1)
        X2 = paddle.concat(x=[M1, M3], axis=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2


class QANet(paddle.nn.Layer):

    def __init__(self, word_mat, char_mat, c_max_len, q_max_len, d_model,
        train_cemb=False, pad=0, dropout=0.1, num_head=1):
        super().__init__()

        if train_cemb:
            self.char_emb = paddle.nn.Embedding(num_embeddings=char_mat.shape[0], 
                                                        embedding_dim=char_mat.shape[1])
            self.char_emb.weight.set_value(char_mat)
            self.char_emb.weight.stop_gradient = False
        else:
            self.char_emb = paddle.nn.Embedding(num_embeddings=char_mat.shape[0], 
                                                        embedding_dim=char_mat.shape[1])
            self.char_emb.weight.set_value(char_mat)
            self.char_emb.weight.stop_gradient = True
        self.word_emb = paddle.nn.Embedding(num_embeddings=word_mat.shape[0], 
                                                        embedding_dim=word_mat.shape[1])
        self.word_emb.weight.set_value(word_mat)
        self.word_emb.weight.stop_gradient = True
        wemb_dim = word_mat.shape[1]
        cemb_dim = char_mat.shape[1]
        self.emb = Embedding(wemb_dim, cemb_dim, d_model)
        self.num_head = num_head
        self.emb_enc = EncoderBlock(conv_num=4, d_model=d_model, num_head=
            num_head, k=7, dropout=0.1)
        self.cq_att = CQAttention(d_model=d_model)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = paddle.nn.LayerList(sublayers=[EncoderBlock(
            conv_num=2, d_model=d_model, num_head=num_head, k=5, dropout=
            0.1) for _ in range(7)])
        self.out = Pointer(d_model)
        self.PAD = pad
        self.Lc = c_max_len
        self.Lq = q_max_len
        self.dropout = dropout

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (paddle.ones_like(x=Cwid) * self.PAD != Cwid).astype(dtype=
            'float32')
        maskQ = (paddle.ones_like(x=Qwid) * self.PAD != Qwid).astype(dtype=
            'float32')
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw, self.Lc), self.emb(Qc, Qw, self.Lq)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = paddle.nn.functional.dropout(x=M0, p=self.dropout, training=
            self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M2 = M0
        M0 = paddle.nn.functional.dropout(x=M0, p=self.dropout, training=
            self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: not p.stop_gradient, self.
            parameters())
        params = sum([np.prod(p.shape) for p in model_parameters])
        print('Trainable parameters:', params)


if __name__ == '__main__':
    paddle.seed(seed=12)
    test_EncoderBlock = False
    test_QANet = True
    test_PosEncoder = False
    if test_EncoderBlock:
        batch_size = 32
        seq_length = 20
        hidden_dim = 96
        x = paddle.rand(shape=[batch_size, seq_length, hidden_dim])
        m = EncoderBlock(4, hidden_dim, 8, 7, seq_length)
        y = m(x, mask=None)
    if test_QANet:
        device = str('cuda:2' if paddle.device.cuda.device_count() >= 1 else
            'cpu').replace('cuda', 'gpu')
        wemb_vocab_size = 5000
        wemb_dim = 300
        cemb_vocab_size = 94
        cemb_dim = 64
        d_model = 96
        batch_size = 32
        q_max_len = 50
        c_max_len = 400
        char_dim = 16
        wv_tensor = paddle.rand(shape=[wemb_vocab_size, wemb_dim])
        cv_tensor = paddle.rand(shape=[cemb_vocab_size, cemb_dim])
        """Class Method: *.random_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        question_lengths = paddle.randint(1, q_max_len, shape=[batch_size], dtype='int64')
        question_wids = paddle.zeros(shape=[batch_size, q_max_len]).astype(
            dtype='int64')
        question_cids = paddle.zeros(shape=[batch_size, q_max_len, char_dim]
            ).astype(dtype='int64')
        """Class Method: *.random_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        context_lengths = paddle.randint(1, c_max_len, shape=[batch_size], dtype='int64')
        context_wids = paddle.zeros(shape=[batch_size, c_max_len]).astype(dtype
            ='int64')
        context_cids = paddle.zeros(shape=[batch_size, c_max_len, char_dim]
            ).astype(dtype='int64')
        for i in range(batch_size):
            question_wids[i, 0:question_lengths[i]] = paddle.randint(1, wemb_vocab_size, shape=[1, question_lengths[i]], dtype='int64')
            question_cids[i, 0:question_lengths[i], :] = paddle.randint(1, cemb_vocab_size, shape=[1, question_lengths[i], char_dim], dtype='int64')
            context_wids[i, 0:context_lengths[i]] = paddle.randint(1, wemb_vocab_size, shape=[1, context_lengths[i]], dtype='int64')
            context_cids[i, 0:context_lengths[i], :] = paddle.randint(1, cemb_vocab_size, shape=[1, context_lengths[i], char_dim], dtype='int64')

        num_head = 1
        qanet = QANet(wv_tensor, cv_tensor, c_max_len, q_max_len, d_model, train_cemb=False, num_head=num_head)
        p1, p2 = qanet(context_wids, context_cids, question_wids, question_cids)
        print(p1.shape)
        print(p2.shape)
    if test_PosEncoder:
        m = PositionalEncoding(d_model=6, max_len=10, dropout=0)
        input = paddle.randn(shape=[3, 10, 6])
        output = m(input)
        print(output)
        output2 = PosEncoder(input)
        print(output2)
