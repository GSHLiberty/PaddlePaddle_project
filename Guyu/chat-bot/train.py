import paddle
import argparse, os
import random
import sys
sys.path.append('../')
from biglm import BIGLM
from data import Vocab
from data_ import DataLoader, s2xy
from optim import Optim
from paddle.distributed.communication.reduce import ReduceOp

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--min_occur_cnt', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--smoothing', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_len_x', type=int)
    parser.add_argument('--min_len_x', type=int)
    parser.add_argument('--max_len_y', type=int)
    parser.add_argument('--min_len_y', type=int)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--approx', type=str, default='none')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)
    parser.add_argument('--backend', type=str)
    return parser.parse_args()


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def average_gradients(model):
    """ Gradient averaging. """
    normal = True
    size = float(paddle.distributed.get_world_size())
# >>>>>>    size = float(torch.distributed.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            paddle.distributed.all_reduce(param.grad.data, op=ReduceOp.SUM)
# >>>>>>            torch.distributed.all_reduce(param.grad.data, op=paddle.
#                 distributed.ReduceOp.SUM)
            param.grad.data /= size
        else:
            normal = False
            break
    return normal


def eval_epoch(lm_args, model, lm_vocab, local_rank, label, batch_acm):
    ds = []
    with open(lm_args.dev_data, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ds.append(line)
    batch_size = 10
    batches = round(len(ds) / batch_size)
    idx = 0
    avg_nll = 0.0
    avg_ppl = 0.0
    avg_acc = 0.0
    count_bsz = 0.0
    count_tok = 0.0
    while idx < len(ds):
        cplb = ds[idx:idx + batch_size]
        ys_truth, ys_inp, msk = s2xy(cplb, lm_vocab, lm_args.max_len_x,
            lm_args.min_len_x, lm_args.max_len_y, lm_args.min_len_y)
        ys_truth = ys_truth
        ys_inp = ys_inp
        msk = msk
        acc, nll, ppl, toks, bsz = model.ppl(ys_truth, ys_inp, msk)
        avg_acc += acc
        avg_nll += nll
        avg_ppl += ppl
        count_bsz += bsz
        count_tok += toks
        idx += batch_size
    print(
        'validating: label %s, batch_acm %d, acc %.6f, nll %.6f, ppl %.6f' %
        (label, batch_acm, avg_acc / count_tok, avg_nll / count_bsz, 
        avg_ppl / count_bsz), flush=True)


def run(args, local_rank):
    """ Distributed Synchronous """
    paddle.seed(seed=1234)
    vocab = Vocab(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    if args.world_size == 1 or paddle.distributed.get_rank() == 0:
        print('vocab.size = %d' % vocab.size, flush=True)
    model = BIGLM(local_rank, vocab, args.embed_dim, args.ff_embed_dim,
        args.num_heads, args.dropout, args.layers, args.smoothing, args.approx)
    if args.start_from is not None:
        ckpt = paddle.load(path=args.start_from)
        model.set_state_dict(state_dict=ckpt['model'])
    model = model
    if args.world_size > 1:
        paddle.seed(seed=1234 + paddle.distributed.get_rank())
        random.seed(5678 + paddle.distributed.get_rank())
    optimizer = Optim(model.embed_dim, args.lr, args.warmup_steps, paddle.
        optimizer.Adam(parameters=model.parameters(), learning_rate=0,
        epsilon=1e-09, beta1=(0.9, 0.998)[0], beta2=(0.9, 0.998)[1],
        weight_decay=0.0))
    train_data = DataLoader(vocab, args.train_data, args.batch_size, args.
        max_len_x, args.min_len_x, args.max_len_y, args.min_len_y)
    batch_acm = 0
    acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    while True:
        if train_data.epoch_id > args.epoch:
            break
        model.train()
        for truth, inp, msk in train_data:
            batch_acm += 1
            truth = truth
            inp = inp
            msk = msk
            model.clear_gradients()
#             """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>            model.zero_grad()
            res, loss, acc, nll, ppl, ntokens, npairs = model(truth, inp, msk)
            loss_acm += loss.item()
            acc_acm += acc
            nll_acm += nll
            ppl_acm += ppl
            ntokens_acm += ntokens
            npairs_acm += npairs
            nxs += npairs
            loss.backward()
            if args.world_size > 1:
                average_gradients(model)
            paddle.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                max_norm=1.0)
            optimizer.step()
            if (args.world_size == 1 or paddle.distributed.get_rank() == 0
                ) and batch_acm % args.print_every == -1 % args.print_every:
                print(
                    'batch_acm %d, loss %.3f, acc %.3f, nll %.3f, ppl %.3f, x_acm %d, lr %.6f'
                     % (batch_acm, loss_acm / args.print_every, acc_acm /
                    ntokens_acm, nll_acm / nxs, ppl_acm / nxs, npairs_acm,
                    optimizer._rate), flush=True)
                acc_acm, nll_acm, ppl_acm, ntokens_acm, loss_acm, nxs = (
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            if (args.world_size == 1 or paddle.distributed.get_rank() == 0
                ) and batch_acm % args.save_every == -1 % args.save_every:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                paddle.save(obj={'args': args, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, path=
                    '%s/epoch%d_batch_%d' % (args.save_dir, train_data.
                    epoch_id, batch_acm))
                model.eval()
                eval_epoch(args, model, vocab, local_rank, 'epoch-' + str(
                    train_data.epoch_id) + '-acm-' + str(batch_acm), batch_acm)
                model.train()


def init_processes(args, local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    paddle.distributed.init_parallel_env()
    fn(args, local_rank)


if __name__ == '__main__':
# >>>>>>    torch.multiprocessing.set_start_method('spawn')
    paddle.set_device('gpu')
    args = parse_config()
    if args.world_size == 1:
        run(args, 0)
        exit(0)
#     processes = []
#     for rank in range(args.gpus):
# >>>>>>        p = torch.multiprocessing.Process(target=init_processes, args=(args,
#             rank, run, args.backend))
#         """Class Method: *.start, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
