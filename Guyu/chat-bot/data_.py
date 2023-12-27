import paddle
import random
import numpy as np
PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
BUFSIZE = 100000


def ListsToTensor(xs, vocab=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if vocab is not None:
            y = vocab.token2idx(x) + [vocab.padding_idx] * (max_len - len(x))
        else:
            y = x + [0] * (max_len - len(x))
        ys.append(y)
    return ys


def batchify(data, vocab):
    truth, inp, msk = [], [], []
    for x, y in data:
        inp.append(x + y[:-1])
        truth.append(x[1:] + y)
        msk.append([(0) for i in range(len(x) - 1)] + [(1) for i in range(
            len(y))])
    
    truth = paddle.to_tensor(data=ListsToTensor(truth, vocab), dtype='int64')
    truth = paddle.transpose(truth, [1, 0])
#     """Class Method: *.t_, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>    truth = paddle.to_tensor(data=ListsToTensor(truth, vocab), dtype='int64'
#         ).t_()
    inp = paddle.to_tensor(data=ListsToTensor(inp, vocab), dtype='int64')
    inp = paddle.transpose(inp, [1, 0])
#     """Class Method: *.t_, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>    inp = paddle.to_tensor(data=ListsToTensor(inp, vocab), dtype='int64').t_()
    msk = paddle.to_tensor(data=ListsToTensor(msk), dtype='float32')
    msk = paddle.transpose(msk, [1, 0])
#     """Class Method: *.t_, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>    msk = paddle.to_tensor(data=ListsToTensor(msk), dtype='float32').t_()
    return truth, inp, msk


def s2t(strs, vocab):
    inp, msk = [], []
    for x in strs:
        inp.append(x)
        msk.append([(1) for i in range(len(x))])
    
    inp = paddle.to_tensor(data=ListsToTensor(inp, vocab), dtype='int64')
    inp = paddle.transpose(inp, [1, 0])
#     """Class Method: *.t_, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>    inp = paddle.to_tensor(data=ListsToTensor(inp, vocab), dtype='int64').t_()
    msk = paddle.to_tensor(data=ListsToTensor(msk), dtype='float32')
    msk = paddle.transpose(msk, [1, 0])
#     """Class Method: *.t_, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>    msk = paddle.to_tensor(data=ListsToTensor(msk), dtype='float32').t_()
    return inp, msk


def s2xy(lines, vocab, max_len_x, min_len_x, max_len_y, min_len_y):
    data = parse_lines(lines, max_len_x, min_len_x, max_len_y, min_len_y)
    return batchify(data, vocab)


def parse_lines(lines, max_len_x, min_len_x, max_len_y, min_len_y):
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        fs = line.split('\t')
        if len(fs) != 2:
            continue
        tokens_x = [w for w in fs[0]]
        tokens_y = [w for w in fs[1]]
        if len(tokens_x) > max_len_x:
            tokens_x = tokens_x[:max_len_x]
        if len(tokens_x) < min_len_x:
            continue
        if len(tokens_y) > max_len_y:
            tokens_y = tokens_y[:max_len_y]
        if len(tokens_y) < min_len_y:
            continue
        data.append((tokens_x + [BOS], tokens_y + [EOS]))
    return data


class DataLoader(object):

    def __init__(self, vocab, filename, batch_size, max_len_x, min_len_x,
        max_len_y, min_len_y):
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len_x = max_len_x
        self.min_len_x = min_len_x
        self.max_len_y = max_len_y
        self.min_len_y = min_len_y
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        lines = self.stream.readlines(BUFSIZE)
        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)
        data = parse_lines(lines[:-1], self.max_len_x, self.min_len_x, self
            .max_len_y, self.min_len_y)
        random.shuffle(data)
        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx + self.batch_size], self.vocab)
            idx += self.batch_size
