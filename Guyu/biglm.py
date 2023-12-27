import sys
sys.path.append('/data2/gsh/paddlepaddle/Guyu/utils')
import paddle
from utils import gelu, LayerNorm
from transformer_postln import TransformerLayer, Embedding, LearnedPositionalEmbedding, SelfAttentionMask
from label_smoothing import LabelSmoothing


class BIGLM(paddle.nn.Layer):

    def __init__(self, local_rank, vocab, embed_dim, ff_embed_dim,
        num_heads, dropout, layers, smoothing_factor, approx):
        super(BIGLM, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.tok_embed = Embedding(self.vocab.size, embed_dim, self.vocab.
            padding_idx)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=
            local_rank)
        self.layers = paddle.nn.LayerList()
        for i in range(layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim,
                num_heads, dropout))
        self.emb_layer_norm = LayerNorm(embed_dim)
        self.one_more = paddle.nn.Linear(in_features=embed_dim,
            out_features=embed_dim)
        self.one_more_layer_norm = LayerNorm(embed_dim)
        self.out_proj = paddle.nn.Linear(in_features=embed_dim,
            out_features=self.vocab.size)
        self.attn_mask = SelfAttentionMask(device=local_rank)
        self.smoothing = LabelSmoothing(local_rank, self.vocab.size, self.
            vocab.padding_idx, smoothing_factor)
        self.dropout = dropout
        self.device = local_rank
        if approx == 'none':
            self.approx = None
        else:
            raise NotImplementedError('%s has not been implemented' % approx)
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.one_more.bias)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.one_more.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.out_proj.bias)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.out_proj.weight)

    def label_smotthing_loss(self, y_pred, y, y_mask, avg=True):
        seq_len, bsz = y.shape
        y_pred = paddle.log(x=y_pred.clip(min=1e-08))
        loss = self.smoothing(paddle.reshape(y_pred, shape=[seq_len * bsz, -1]), paddle.reshape(y, shape=[seq_len * bsz, -1]))
        if avg:
            return loss / paddle.sum(x=y_mask)
        else:
            return loss / bsz

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -paddle.log(x=paddle.take_along_axis(arr=y_pred, axis=2, indices=paddle.reshape(y, shape=[y.shape[0], y.shape[1], 1])))
        cost = paddle.reshape(cost, shape=y.shape)
        y_mask = paddle.reshape(y_mask, shape=y.shape)
        if avg:
            cost = paddle.sum(x=cost * y_mask, axis=0) / paddle.sum(x=
                y_mask, axis=0)
        else:
            cost = paddle.sum(x=cost * y_mask, axis=0)
        cost = paddle.reshape(cost, shape=[y.shape[1], -1])
        ppl = 2 ** cost
        return cost.sum().item(), ppl.sum().item()

    def ppl(self, truth, inp, msk):
        seq_len, bsz = inp.shape
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        padding_mask = paddle.equal(x=truth, y=self.vocab.padding_idx)
        if not padding_mask.astype('bool').any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = paddle.nn.functional.softmax(x=self.out_proj(x), axis=-1)
        _, pred_y = pred.max(-1)
        tot_tokens = msk.astype(dtype='float32').sum().item()
        acc = (paddle.equal(x=pred_y, y=truth).astype(dtype='float32') * msk
            ).sum().item()
        nll, ppl = self.nll_loss(pred, truth, msk)
        return acc, nll, ppl, tot_tokens, bsz

    def work(self, inp):
        seq_len, bsz = inp.shape
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self
            .training)
        padding_mask = paddle.equal(x=inp, y=self.vocab.padding_idx)
        if not padding_mask.astype('bool').any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        probs = paddle.nn.functional.softmax(x=self.out_proj(x), axis=-1)
        _, pred_y = probs.max(-1)
        return probs, pred_y

    def work_incremental(self, inp, incremental_state=None):
        seq_len, bsz = inp.shape
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        padding_mask = paddle.equal(x=inp, y=self.vocab.padding_idx)
        if not padding_mask.astype('bool').any():
            padding_mask = None
        if incremental_state is None:
            self_attn_mask = self.attn_mask(seq_len)
            incremental_state = {}
        else:
            x = x[-1, :, :].unsqueeze(axis=0)
            self_attn_mask = None
        for layer in self.layers:
            x, _, _ = layer.work_incremental(x, self_padding_mask=
                padding_mask, self_attn_mask=self_attn_mask,
                incremental_state=incremental_state)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        probs = paddle.nn.functional.softmax(x=self.out_proj(x), axis=-1)
        _, pred_y = probs.max(-1)
        return probs, pred_y, incremental_state

    def forward(self, truth, inp, msk):
        seq_len, bsz = inp.shape
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self
            .training)
        padding_mask = paddle.equal(x=truth, y=self.vocab.padding_idx)
        if not padding_mask.astype('bool').any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = paddle.nn.functional.softmax(x=self.out_proj(x), axis=-1)
        loss = self.label_smotthing_loss(pred, truth, msk)
        _, pred_y = pred.max(-1)
        tot_tokens = msk.astype(dtype='float32').sum().item()
        acc = (paddle.equal(x=pred_y, y=truth).astype(dtype='float32') * msk
            ).sum().item()
        nll, ppl = self.nll_loss(pred, truth, msk)
        return (pred_y, truth), loss, acc, nll, ppl, tot_tokens, bsz
