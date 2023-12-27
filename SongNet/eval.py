import paddle
import sys
import random
import numpy as np
import copy
import time
from biglm import BIGLM
from data import Vocab, DataLoader, s2t, s2xy
gpu = int(sys.argv[2]) if len(sys.argv) > 1 else 0


def init_model(m_path, device, vocab):
    ckpt = paddle.load(path=m_path)
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.
        ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 
        0.1, lm_args.approx)
    lm_model.set_state_dict(state_dict=ckpt['model'])
    lm_model = lm_model
    lm_model.eval()
    return lm_model, lm_vocab, lm_args


m_path = sys.argv[1] if len(sys.argv) > 1 else None
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, './data/vocab.txt')
ds = []
with open('./data/dev.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            ds.append(line)
print(len(ds))
local_rank = gpu
batch_size = 10
batches = round(len(ds) / batch_size)
idx = 0
avg_nll = 0.0
avg_ppl = 0.0
count = 0.0
while idx < len(ds):
    cplb = ds[idx:idx + batch_size]
    (xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk
        ) = s2xy(cplb, lm_vocab, lm_args.max_len, 2)
    xs_tpl = xs_tpl
    xs_seg = xs_seg
    xs_pos = xs_pos
    ys_truth = ys_truth
    ys_inp = ys_inp
    ys_tpl = ys_tpl
    ys_seg = ys_seg
    ys_pos = ys_pos
    msk = msk
    nll, ppl, bsz = lm_model.ppl(xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp,
        ys_tpl, ys_seg, ys_pos, msk)
    avg_nll += nll
    avg_ppl += ppl
    count += bsz
    idx += batch_size
    if count % 200 == 0:
        print('nll=', avg_nll / count, 'ppl=', avg_ppl / count, 'count=', count
            )
print('nll=', avg_nll / count, 'ppl=', avg_ppl / count, 'count=', count)
