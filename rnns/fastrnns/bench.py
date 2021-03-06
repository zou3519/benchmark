import argparse
from collections import namedtuple
import torch
import gc
import sys
import json

from .runner import get_rnn_runners


BenchResult = namedtuple('BenchResult', [
    'name', 'avg_fwd', 'std_fwd', 'avg_bwd', 'std_bwd',
])


def fit_str(string, colwidth=16):
    if len(string) < colwidth:
        return (colwidth - len(string)) * ' ' + string
    else:
        return string[:colwidth]


def to_str(item):
    if isinstance(item, float):
        return '%.4g' % item
    return str(item)


def print_header(colwidth=16, sep=' '):
    items = []
    for item in BenchResult._fields:
        items.append(fit_str(item))
    return sep.join(items)


def pretty_print(benchresult, colwidth=16, sep=' '):
    items = []
    for thing in benchresult:
        items.append(fit_str(to_str(thing)))
    return sep.join(items)


def trainbench(name, rnn_creator, nloops=100, warmup=10,
               seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
               miniBatch=64, device='cuda', seed=None):
    def train_batch(rnn, inputs, params):
        # CUDA events for timing
        fwd_start_event = torch.cuda.Event(enable_timing=True)
        fwd_end_event = torch.cuda.Event(enable_timing=True)
        bwd_start_event = torch.cuda.Event(enable_timing=True)
        bwd_end_event = torch.cuda.Event(enable_timing=True)

        gc.collect()

        fwd_start_event.record()
        output = rnn(*inputs)
        fwd_end_event.record()
        if isinstance(output, tuple):
            output = output[0]

        grads = torch.rand_like(output)
        gc.collect()

        bwd_start_event.record()
        output.backward(grads)
        bwd_end_event.record()

        for param in params:
            param.grad.data.zero_()

        torch.cuda.synchronize()

        fwd_time = fwd_start_event.elapsed_time(fwd_end_event)
        bwd_time = bwd_start_event.elapsed_time(bwd_end_event)
        return fwd_time, bwd_time

    assert device == 'cuda'
    creator_args = dict(seqLength=seqLength, numLayers=numLayers,
                        inputSize=inputSize, hiddenSize=hiddenSize,
                        miniBatch=miniBatch, device=device, seed=seed)
    rnn, inputs, params = rnn_creator(**creator_args)

    [train_batch(rnn, inputs, params) for _ in range(warmup)]

    results = [train_batch(rnn, inputs, params) for _ in range(nloops)]
    fwd_times, bwd_times = zip(*results)

    fwd_times = torch.tensor(fwd_times)
    bwd_times = torch.tensor(bwd_times)

    return BenchResult(name=name,
                       avg_fwd=fwd_times.mean().item(),
                       std_fwd=fwd_times.std().item(),
                       avg_bwd=bwd_times.mean().item(),
                       std_bwd=bwd_times.std().item())


def print_stderr(*args, **kwargs):
    return print(*args, **kwargs, file=sys.stderr)


def bench(rnn_runners, group_name, print_json=False, sep=' ', **params):
    print_stderr(print_header(sep=sep))
    results = {}
    for name, creator, context in rnn_runners:
        with context():
            try:
                result = trainbench(name, creator, **params)
                print_stderr(pretty_print(result, sep=sep))
                results[name] = result
            except Exception as e:
                if not print_json:
                    raise

    return {
        group_name: {k: v.avg_fwd for k, v in results.items()},
        group_name + '-backward': {k: v.avg_bwd for k, v in results.items()},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile RNNs')

    parser.add_argument('--seqLength', default='100', type=int)
    parser.add_argument('--numLayers', default='1', type=int)
    parser.add_argument('--inputSize', default='512', type=int)
    parser.add_argument('--hiddenSize', default='512', type=int)
    parser.add_argument('--miniBatch', default='64', type=int)
    parser.add_argument('--warmup', default='10', type=int)
    parser.add_argument('--nloops', default='100', type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--sep', default=' ', type=str)
    parser.add_argument('--print-json', action='store_true')
    parser.add_argument('--rnns', nargs='*',
                        help='What to run. cudnn, aten, jit, etc')

    args = parser.parse_args()
    rnns = args.rnns or ['cudnn', 'aten', 'jit', 'jit_premul', 'jit_simple', 'jit_multilayer', 'py']
    cnns = ['resnet18', 'resnet18_jit', 'resnet50', 'resnet50_jit']
    if args.print_json:
        print_stderr = lambda *args, **kwargs: None
    print_stderr(args)

    bench_args = vars(args)
    del bench_args['rnns']

    print_stderr('Benchmarking LSTMs...')
    rnn_results = bench(get_rnn_runners(*rnns), 'lstm', **bench_args)
    print_stderr('')

    print_stderr('Benchmarking ResNets...')
    cnn_results = bench(get_rnn_runners(*cnns), 'resnet', **bench_args)
    print_stderr('')

    if args.print_json:
        rnn_results.update(cnn_results)
        print(json.dumps(rnn_results))
