import paddle
from biglm import BIGLM
from data import Vocab, DataLoader, s2xy
from optim import Optim
import argparse, os
import random
from paddle.distributed.communication.reduce import ReduceOp
import paddle.distributed as dist

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
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--min_len', type=int)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--save_dir', type=str)
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
    for param in model.parameters():
        if param.grad is not None:
            paddle.distributed.all_reduce(param.grad.data, op=ReduceOp.SUM)
            param.grad.data /= size
        else:
            normal = False
            break
    return normal


def eval_epoch(lm_args, model, lm_vocab, local_rank, label):
    print('validating...', flush=True)
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
    count = 0.0
    while idx < len(ds):
        cplb = ds[idx:idx + batch_size]
        (xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk
            ) = s2xy(cplb, lm_vocab, lm_args.max_len, lm_args.min_len)
        xs_tpl = xs_tpl
        xs_seg = xs_seg
        xs_pos = xs_pos
        ys_truth = ys_truth
        ys_inp = ys_inp
        ys_tpl = ys_tpl
        ys_seg = ys_seg
        ys_pos = ys_pos
        msk = msk
        nll, ppl, bsz = model.ppl(xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp,
            ys_tpl, ys_seg, ys_pos, msk)
        avg_nll += nll
        avg_ppl += ppl
        count += bsz
        idx += batch_size
    print(label, 'nll=', avg_nll / count, 'ppl=', avg_ppl / count, 'count=',
        count, flush=True)


def run(args, local_rank):
    """ Distributed Synchronous """
    paddle.seed(seed=1234)
    vocab = Vocab(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    if args.world_size == 1 or paddle.distributed.get_rank() == 0:
        print('vocab.size = ' + str(vocab.size), flush=True)
    model = BIGLM(local_rank, vocab, args.embed_dim, args.ff_embed_dim,
        args.num_heads, args.dropout, args.layers, args.smoothing)
    if args.start_from is not None:
        ckpt = paddle.load(path=args.start_from)
        model.set_state_dict(state_dict=ckpt['model'])
    model = model
    optimizer = Optim(model.embed_dim, args.lr, args.warmup_steps, paddle.
        optimizer.Adam(parameters=model.parameters(), learning_rate=0.0,
        epsilon=1e-09, beta1=(0.9, 0.998)[0], beta2=(0.9, 0.998)[1],
        weight_decay=0.0))
    if args.start_from is not None:
        optimizer.set_state_dict(state_dict=ckpt['optimizer'])
    train_data = DataLoader(vocab, args.train_data, args.batch_size, args.
        max_len, args.min_len)
    batch_acm = 0
    acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    while True:
        model.train()
        if train_data.epoch_id > 30:
            break
        for xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk in train_data:
            batch_acm += 1
            xs_tpl = xs_tpl
            xs_seg = xs_seg
            xs_pos = xs_pos
            ys_truth = ys_truth
            ys_inp = ys_inp
            ys_tpl = ys_tpl
            ys_seg = ys_seg
            ys_pos = ys_pos
            msk = msk
            model.clear_gradients()
            res, loss, acc, nll, ppl, ntokens, npairs = model(xs_tpl,
                xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk)
            loss_acm += loss.item()
            acc_acm += acc
            nll_acm += nll
            ppl_acm += ppl
            ntokens_acm += ntokens
            npairs_acm += npairs
            nxs += npairs
            loss.backward()
            if args.world_size > 1:
                is_normal = average_gradients(model)
            else:
                is_normal = True
            if is_normal:
                parameters_to_clip = [p for p in model.parameters() if p.grad is not None and p.grad.shape != []]
                # 对这些参数进行梯度裁剪
                # parameters_to_clip.astype('float32')
                paddle.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=1.0)
                optimizer.step()
            else:
                print('gradient: none, gpu: ' + str(local_rank), flush=True)
                continue
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
                model.eval()
                eval_epoch(args, model, vocab, local_rank, 'epoch-' + str(
                    train_data.epoch_id) + '-acm-' + str(batch_acm))
                model.train()
                paddle.save(obj={'args': args, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, path=
                    '%s/epoch%d_batch_%d' % (args.save_dir, train_data.
                    epoch_id, batch_acm))


def init_processes(args, local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    paddle.distributed.init_parallel_env()
    fn(args, local_rank)


if __name__ == '__main__':
# >>>>>>    torch.multiprocessing.set_start_method('spawn')
    # dist.spawn(train)
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
