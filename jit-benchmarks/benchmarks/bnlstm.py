import torch
from torch import nn
from torch.autograd import Variable
from . import *
import argparse

import models.bnlstm as bnlstm
from common import AttrDict, Bench

# From https://github.com/jihunchoi/recurrent-batch-normalization-pytorch


def run_bnlstm(args):
    default_params = dict(hidden_size=100, max_length=784, pmnist=False, num_batches=1, cuda=False)
    params = AttrDict(default_params)
    p = params

    bench = Bench()

    # The CPU version is slow...
    p['batch_size'] = 20 if p.cuda else 5
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.rnn = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell, input_size=1, hidden_size=p.hidden_size, batch_first=True, max_length=p.max_length)
            self.fc = nn.Linear(in_features=p.hidden_size, out_features=10) # 10 digits in mnist

        def forward(self, data):
            hx = None
            if not p.pmnist:
                h0 = Variable(data.data.new(data.size(0), p.hidden_size)
                              .normal_(0, 0.1))
                c0 = Variable(data.data.new(data.size(0), p.hidden_size)
                              .normal_(0, 0.1))
                hx = (h0, c0)
            _, (h_n, _) = self.rnn(input_=data, hx = hx)
            logits = self.fc(h_n[0])
            return logits

    def cast(tensor):
        return tensor.cuda() if p.cuda else tensor

    model = Model()
    criterion = nn.CrossEntropyLoss()
    data_batches = [Variable(cast(torch.zeros(p.batch_size, 28 * 28, 1))) for _ in range(p.num_batches)]
    target_batches = [Variable(cast(torch.zeros(p.batch_size)).long()) for _ in range(p.num_batches)]
    if p.cuda:
        model.cuda()
        criterion.cuda()

    total_loss = 0
    for data, targets in zip(data_batches, target_batches):
        bench.start_timing()
        logits = model(data)
        loss = criterion(input=logits, target=targets)
        loss.backward()
        bench.stop_timing()
        total_loss += loss.data  # CUDA sync point
    if p.cuda:
        torch.cuda.synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch bnlstm bench")
    parser.add_argument('--warmup',     type=int, default=2,   help="Warmup iterations")
    parser.add_argument('--benchmark',  type=int, default=10,  help="Benchmark iterations")
    parser.add_argument('--jit',        action='store_true',   help="Use JIT compiler")
    args = parser.parse_args()

    run_bnlstm(args)
