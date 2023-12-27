import sys
sys.path.append('/data2/gsh/paddlepaddle/Guyu/utils')
import paddle_aux
import paddle
from utils import gelu, LayerNorm, get_incremental_state, set_incremental_state
import math


class TransformerLayer(paddle.nn.Layer):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout,
        with_external=False, weights_dropout=True):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout,
            weights_dropout)
        self.fc1 = paddle.nn.Linear(in_features=embed_dim, out_features=
            ff_embed_dim)
        self.fc2 = paddle.nn.Linear(in_features=ff_embed_dim, out_features=
            embed_dim)
        self.attn_layer_norm = LayerNorm(embed_dim)
        self.ff_layer_norm = LayerNorm(embed_dim)
        self.with_external = with_external
        self.dropout = dropout
        if self.with_external:
            self.external_attn = MultiheadAttention(embed_dim, num_heads,
                dropout, weights_dropout)
            self.external_layer_norm = LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.fc1.weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.fc2.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.fc1.bias)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.fc2.bias)

    def forward(self, x, kv=None, self_padding_mask=None, self_attn_mask=
        None, external_memories=None, external_padding_mask=None,
        need_weights=False):
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(query=x, key=x, value=x,
                key_padding_mask=self_padding_mask, attn_mask=
                self_attn_mask, need_weights=need_weights)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv,
                key_padding_mask=self_padding_mask, attn_mask=
                self_attn_mask, need_weights=need_weights)
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self
            .training)
        x = self.attn_layer_norm(residual + x)
        if self.with_external:
            residual = x
            x, external_attn = self.external_attn(query=x, key=
                external_memories, value=external_memories,
                key_padding_mask=external_padding_mask, need_weights=
                need_weights)
            x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=
                self.training)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None
        residual = x
        x = gelu(self.fc1(x))
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self
            .training)
        x = self.fc2(x)
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self
            .training)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn, external_attn

    def work_incremental(self, x, self_padding_mask, self_attn_mask,
        incremental_state):
        residual = x
        x, self_attn = self.self_attn(query=x, key=x, value=x,
            key_padding_mask=self_padding_mask, attn_mask=self_attn_mask,
            incremental_state=incremental_state)
        x = self.attn_layer_norm(residual + x)
        residual = x
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn, None


class MultiheadAttention(paddle.nn.Layer):

    def __init__(self, embed_dim, num_heads, dropout=0.0, weights_dropout=True
        ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        out_0 = paddle.create_parameter(shape=paddle.empty(shape=[3 * embed_dim, embed_dim]).shape, 
                                        dtype='float32',
                                        # dtype=paddle.empty(shape=[3 *embed_dim, embed_dim]).numpy().dtype, 
                                        default_initializer=paddle.nn.initializer.Assign(paddle.empty(shape=[3 * embed_dim, embed_dim])))
        out_0.stop_gradient = not True
        self.in_proj_weight = out_0
        out_1 = paddle.create_parameter(shape = [3 * embed_dim],
                                        # shape=paddle.to_tensor(data=3 * embed_dim, dtype='float32').shape, 
                                        dtype='float32',
                                        default_initializer=paddle.nn.initializer.Assign(paddle.empty(shape=[3 * embed_dim])))
                                        # dtype=paddle.to_tensor(data=3 * embed_dim, dtype='float32').numpy().dtype,
                                        # default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(data=3 * embed_dim, dtype='float32')))
        out_1.stop_gradient = not True
        self.in_proj_bias = out_1
        self.out_proj = paddle.nn.Linear(in_features=embed_dim,
            out_features=embed_dim, bias_attr=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.in_proj_weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.out_proj.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.in_proj_bias)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.out_proj.bias)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=
        None, need_weights=False, incremental_state=None):
        """ Input shape: Time x Batch x Channel
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            bidx = self._get_bidx(incremental_state)
        else:
            saved_state = None
            bidx = None

        qkv_same = (query.shape == key.shape) and (query.shape == value.shape) and (query.dtype == value.dtype) and (query.dtype == key.dtype) and paddle.all(paddle.equal(query, key)) and paddle.all(paddle.equal(query, value))
        kv_same = (key.shape == value.shape) and (key.dtype == value.dtype) and paddle.all(paddle.equal(key, value))
        tgt_len, bsz, embed_dim = query.shape
        assert key.shape == value.shape
        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling
        x = paddle.reshape(q, shape=[tgt_len, bsz * self.num_heads, self.head_dim])
        perm_0 = list(range(x.ndim))
        perm_0[0] = 1
        perm_0[1] = 0
        q = x.transpose(perm=perm_0)
        x = paddle.reshape(k, shape=[-1, bsz * self.num_heads, self.head_dim])
        perm_1 = list(range(x.ndim))
        perm_1[0] = 1
        perm_1[1] = 0
        k = x.transpose(perm=perm_1)
        x = paddle.reshape(v, shape=[-1, bsz * self.num_heads, self.head_dim])
        perm_2 = list(range(x.ndim))
        perm_2[0] = 1
        perm_2[1] = 0
        v = x.transpose(perm=perm_2)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key']
                if bidx is not None:
                    prev_key = prev_key[bidx]
                prev_key = paddle.reshape(prev_key, shape=[bsz * self.num_heads, -1, self.head_dim])
                k = paddle.concat(x=(prev_key, k), axis=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value']
                if bidx is not None:
                    prev_value = prev_value[bidx]
                prev_value = paddle.reshape(prev_value, shape=[bsz * self.num_heads, -1, self.head_dim])
                v = paddle.concat(x=(prev_value, v), axis=1)
            saved_state['prev_key'] = paddle.reshape(k, shape=[bsz, self.num_heads, -1, self.head_dim])
            saved_state['prev_value'] = paddle.reshape(v, shape=[bsz, self.num_heads, -1, self.head_dim])
            self._set_input_buffer(incremental_state, saved_state)
        src_len = k.shape[1]
        x = k
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        attn_weights = paddle.bmm(x=q, y=x.transpose(perm=perm_3))
        assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len,
            src_len]
        if attn_mask is not None:
            attn_weights = paddle.where(attn_mask.unsqueeze(axis=0), paddle.full_like(attn_weights, float('-inf')), attn_weights)
        if key_padding_mask is not None:
            attn_weights = paddle.reshape(attn_weights, shape=[bsz, self.num_heads, tgt_len, src_len])

            key_padding_mask_transposed = paddle.transpose(key_padding_mask, perm=[1, 0])
            mask = paddle.unsqueeze(key_padding_mask_transposed, axis=[1, 2])
            attn_weights = paddle.where(mask, paddle.full_like(attn_weights, float('-inf')), attn_weights)
            attn_weights = paddle.reshape(attn_weights, shape=[bsz * self.num_heads, tgt_len, src_len])
        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1)
        if self.weights_dropout:
            attn_weights = paddle.nn.functional.dropout(x=attn_weights, p=
                self.dropout, training=self.training)
        attn = paddle.bmm(x=attn_weights, y=v)
        if not self.weights_dropout:
            attn = paddle.nn.functional.dropout(x=attn, p=self.dropout,
                training=self.training)
        assert list(attn.shape) == [bsz * self.num_heads, tgt_len, self.
            head_dim]
        x = attn
        perm_5 = list(range(x.ndim))
        perm_5[0] = 1
        perm_5[1] = 0
        attn = paddle.reshape(x.transpose(perm=perm_5), shape=[tgt_len, bsz, embed_dim])
        attn = self.out_proj(attn)
        if need_weights:
            attn_weights = paddle.reshape(attn_weights, shape=[bsz, self.num_heads, tgt_len, src_len])
            attn_weights, _ = attn_weights.max(dim=1)
            x = attn_weights
            perm_6 = list(range(x.ndim))
            perm_6[0] = 1
            perm_6[1] = 0
            attn_weights = x.transpose(perm=perm_6)
        else:
            attn_weights = None
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(chunks=3, axis=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(chunks=2, axis=-1
            )

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return paddle.nn.functional.linear(weight=weight.T, bias=bias, x=input)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(self, incremental_state, 'attn_state'
            ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(self, incremental_state, 'attn_state', buffer)

    def _get_bidx(self, incremental_state):
        if 'bidx' in incremental_state:
            return incremental_state['bidx']
        else:
            return None


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = paddle.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=
        embedding_dim, padding_idx=padding_idx)
    init_Normal = paddle.nn.initializer.Normal(std=0.02)
    init_Normal(m.weight)
    init_Constant = paddle.nn.initializer.Constant(value=0)
    init_Constant(m.weight[padding_idx])
    return m


class SelfAttentionMask(paddle.nn.Layer):

    def __init__(self, init_size=100, device=0):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)
        self.device = device

    @staticmethod
    def get_mask(size):
        weights = paddle.triu(x=paddle.ones(shape=(size, size), dtype=
            'bool'), diagonal=1)
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.shape[0]:
            self.weights = SelfAttentionMask.get_mask(size)
        res = self.weights[:size, :size].to(self.device).detach()
        return res


class LearnedPositionalEmbedding(paddle.nn.Layer):
    """This module produces LearnedPositionalEmbedding.
    """

    def __init__(self, embedding_dim, init_size=1024, device=0):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = paddle.nn.Embedding(num_embeddings=init_size,
            embedding_dim=embedding_dim)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.weights.weight)

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        seq_len, bsz = input.shape
        positions = (offset + paddle.arange(end=seq_len)).to(self.device)
        res = self.weights(positions).unsqueeze(axis=1).expand(shape=[-1,
            bsz, -1])
        return res


class SinusoidalPositionalEmbedding(paddle.nn.Layer):
    """This module produces sinusoidal positional embeddings of any length.
    """

    def __init__(self, embedding_dim, init_size=1024, device=0):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size,
            embedding_dim)
        self.device = device

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(dtype='float32', end=half_dim) * -emb)
        emb = paddle.arange(dtype='float32', end=num_embeddings).unsqueeze(axis
            =1) * emb.unsqueeze(axis=0)
        emb = paddle.reshape(paddle.concat(x=[paddle.sin(x=emb), paddle.cos(x=emb)], axis=1), shape=[num_embeddings, -1])
        if embedding_dim % 2 == 1:
            emb = paddle.concat(x=[emb, paddle.zeros(shape=[num_embeddings,
                1])], axis=1)
        return emb

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        seq_len, bsz = input.shape
        mx_position = seq_len + offset
        if self.weights is None or mx_position > self.weights.shape[0]:
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                mx_position, self.embedding_dim)
        positions = offset + paddle.arange(end=seq_len)
        res = self.weights.index_select(axis=0, index=positions).unsqueeze(axis
            =1).expand(shape=[-1, bsz, -1]).to(self.device).detach()
        return res
