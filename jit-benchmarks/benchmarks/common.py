import time
import sys
from itertools import product
import torch
from collections import namedtuple


PY2 = sys.version_info[0] == 2


class over(object):
    def __init__(self, *args):
        self.values = args


class AttrDict(dict):
    def __repr__(self):
        return ', '.join(k + '=' + str(v) for k, v in self.items())

    def __getattr__(self, name):
        return self[name]


def make_params(**kwargs):
    keys = list(kwargs.keys())
    iterables = [kwargs[k].values if isinstance(kwargs[k], over) else (kwargs[k],) for k in keys]
    all_values = list(product(*iterables))
    param_dicts = [AttrDict({k: v for k, v in zip(keys, values)}) for values in all_values]
    return [param_dicts]


class Benchmark(object):
    timer = time.time if PY2 else time.perf_counter
    default_params = []
    params = make_params()
    param_names = ['config']

    # NOTE: subclasses should call prepare instead of setup
    def setup(self, params):
        for k, v in self.default_params.items():
            params.setdefault(k, v)
        self.prepare(params)


SummaryStats = namedtuple('SummaryStats', ['mean', 'min', 'max'])


class Bench(object):

    def __init__(self, name='bench', cuda=False):
        self.results = []
        self.timing = False
        self.iter = 0

        self.name = name
        self.cuda = cuda

    def start_timing(self):
        assert not self.timing

        self.timing = True
        if self.cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=False)
            self.start.record()
        self.start_cpu_secs = time.time()

    def stop_timing(self):
        assert self.timing

        end_cpu_secs = time.time()
        if self.cuda:
            self.end.record()
            torch.cuda.synchronize()
            gpu_msecs = self.start.elapsed_time(self.end)
        else:
            gpu_msecs = 0
        cpu_msecs = (end_cpu_secs - self.start_cpu_secs) * 1000

        if self.cuda:
            print('%s(%2d) gpu %.3f msecs (cpu %.3f msecs)' %
                  (self.name, self.iter, gpu_msecs, cpu_msecs))
        else:
            print('%s(%2d) cpu %.3f msecs' %
                  (self.name, self.iter, cpu_msecs))

        self.iter += 1
        self.timing = False
        self.start = None
        self.end = None
        self.start_cpu_secs = None
        self.results.append([gpu_msecs, cpu_msecs])

    def summary(self):
        assert not self.timing

        def mean_min_max(lst):
            return SummaryStats(sum(lst)/len(lst), min(lst), max(lst))

        gpu_msecs, cpu_msecs = zip(*self.results)
        return mean_min_max(gpu_msecs), mean_min_max(cpu_msecs)
