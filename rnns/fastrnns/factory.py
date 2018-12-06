import torch

from collections import namedtuple

from .cells import lstm_cell, premul_lstm_cell, flat_lstm_cell
from torch.nn.utils.rnn import pack_sequence


# list[list[T]] -> list[T]
def flatten_list(lst):
    result = []
    for inner in lst:
        result.extend(inner)
    return result

# Define a creator as a function: (sizes) -> (rnn, rnn_inputs, flat_rnn_params)
# rnn: function / graph executor / module
# rnn_inputs: the inputs to the returned 'rnn'
# flat_rnn_params: List[Tensor] all requires_grad=True parameters in a list
# One can call rnn(rnn_inputs) using the outputs of the creator.


def pytorch_lstm_creator(**kwargs):
    input, hidden, _, module = lstm_inputs(return_module=True, **kwargs)
    return module, [input, hidden], flatten_list(module.all_weights)


def lstm_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return lstm_factory(lstm_cell, script), inputs, flatten_list(params)


def lstm_premul_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return lstm_factory_premul(premul_lstm_cell, script), inputs, flatten_list(params)


def lstm_simple_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input] + [h[0] for h in hidden] + params[0]
    return lstm_factory_simple(flat_lstm_cell, script), inputs, flatten_list(params)


def lstm_multilayer_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden, flatten_list(params)]
    return lstm_factory_multilayer(lstm_cell, script), inputs, flatten_list(params)


def imagenet_cnn_creator(arch, jit=True):
    def creator(device='cuda', **kwargs):
        model = arch().to(device)
        x = torch.randn(32, 3, 224, 224, device=device)
        if jit:
            model = torch.jit.trace(model, x)
        return model, (x,), list(model.parameters())

    return creator


# input: lstm.all_weights format (wih, whh, bih, bhh = lstm.all_weights[layer])
# output: packed_weights with format
# packed_weights[0] is wih with size (layer, 4*hiddenSize, inputSize)
# packed_weights[1] is whh with size (layer, 4*hiddenSize, hiddenSize)
# packed_weights[2] is bih with size (layer, 4*hiddenSize)
# packed_weights[3] is bhh with size (layer, 4*hiddenSize)
def stack_weights(weights):
    def unzip_columns(mat):
        assert isinstance(mat, list)
        assert isinstance(mat[0], list)
        layers = len(mat)
        columns = len(mat[0])
        return [[mat[layer][col] for layer in range(layers)]
                for col in range(columns)]

    # XXX: script fns have problems indexing multidim lists, so we try to
    # avoid them by stacking tensors
    all_weights = weights
    packed_weights = [torch.stack(param)
                      for param in unzip_columns(all_weights)]
    return packed_weights


# returns: x, (hx, cx), all_weights, lstm module with all_weights as params
def lstm_inputs(seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
                miniBatch=64, return_module=False, device='cuda', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(seqLength, miniBatch, inputSize, device=device)
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers)
    if 'cuda' in device:
        lstm = lstm.cuda()

    if return_module:
        return x, (hx, cx), lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # wih, whh, bih, bhh = lstm.all_weights[layer]
        return x, (hx, cx), lstm.all_weights, None


def lstm_factory(cell, script):
    def dynamic_rnn(input, hidden, wih, whh, bih, bhh):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden
        outputs = []
        inputs = input.unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], (hy, cy), wih, whh, bih, bhh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn



# premul: we're going to premultiply the inputs & weights
def lstm_factory_premul(premul_cell, script):
    def dynamic_rnn(input, hidden, wih, whh, bih, bhh):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden
        outputs = []
        inputs = torch.matmul(input, wih.t()).unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), whh, bih, bhh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# simple: flat inputs (no tuples), no list to accumulate outputs
#         useful mostly for benchmarking older JIT versions
def lstm_factory_simple(cell, script):
    def dynamic_rnn(input, hx, cx, wih, whh, bih, bhh):
        hy = hx  # for scoping
        cy = cx  # for scoping
        inputs = input.unbind(0)
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], hy, cy, wih, whh, bih, bhh)
        return hy, cy

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def lstm_factory_multilayer(cell, script):
    def dynamic_rnn(input, hidden, params):
        # type: (Tensor, Tuple[Tensor, Tensor], List[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        params_stride = 4  # NB: this assumes that biases are there
        hx, cx = hidden
        hy, cy = hidden  # for scoping...
        inputs, outputs = input.unbind(0), []
        for layer in range(hx.size(0)):
            hy = hx[layer]
            cy = cx[layer]
            base_idx = layer * params_stride
            wih = params[base_idx]
            whh = params[base_idx + 1]
            bih = params[base_idx + 2]
            bhh = params[base_idx + 3]
            for seq_idx in range(len(inputs)):
                hy, cy = cell(inputs[seq_idx], (hy, cy), wih, whh, bih, bhh)
                outputs += [hy]
            inputs, outputs = outputs, []
        return torch.stack(inputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def make_varlen_inputs(num_inputs=64, input_size=512, hidden_size=512,
                       min_seq_len=60, max_seq_len=120, device='cuda', seed=17):
    torch.manual_seed(seed)
    seq_lens = torch.randint(min_seq_len, max_seq_len, [num_inputs]).sort(descending=True)[0]
    hx = torch.randn(1, num_inputs, hidden_size, device=device)
    cx = torch.randn(1, num_inputs, hidden_size, device=device)
    x = []
    for i in range(num_inputs):
        x.append(torch.randn(seq_lens[i], 1, input_size, device=device))
    return x, hx, cx


def make_weights(input_size=512, hidden_size=512, device='cuda',
                 return_module=False):
    lstm = torch.nn.LSTM(input_size, hidden_size, 1)
    if 'cuda' in device:
        lstm = lstm.cuda()

    if return_module:
        return lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # wih, whh, bih, bhh = lstm.all_weights[layer]
        return lstm.all_weights, None


# type: (List[Tensor], Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> List[Tuple[Tensor, Tuple[Tensor, Tensor]]]
def varlen_lstm_factory(cell, script):
    def dynamic_rnn(input, hx, cx, wih, whh, bih, bhh):
        # type: (List[Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> List[Tensor]
        hxs = hx.unbind(1)
        cxs = cx.unbind(1)
        # List of: (output, (hiddens))
        outputs = []

        for batch in range(len(input)):
            output = []
            hy, cy = hxs[batch], cxs[batch]
            inputs = input[batch].unbind(0)

            for seq_idx in range(len(inputs)):
                hy, cy = cell(inputs[seq_idx], (hy, cy), wih, whh, bih, bhh)
                output += [hy]
            outputs.append(torch.stack(output))
            outputs.append(hy.unsqueeze(0))
            outputs.append(cy.unsqueeze(0))

        return outputs

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def varlen_pytorch_lstm_creator(**kwargs):
    x, hx, cx = make_varlen_inputs(
        num_inputs=kwargs['miniBatch'],
        input_size=kwargs['inputSize'],
        min_seq_len=kwargs['seqLength'] - 30,
        max_seq_len=kwargs['seqLength'] + 30,
        device=kwargs['device'],
        seed=17)
    _, module = make_weights(
        input_size=kwargs['inputSize'],
        hidden_size=kwargs['hiddenSize'],
        device=kwargs['device'],
        return_module=True)
    
    inp = pack_sequence([y.squeeze(1) for y in x])
    return module, [inp, (hx, cx)], flatten_list(module.all_weights)


def varlen_lstm_creator(script=True, **kwargs):
    x, hx, cx = make_varlen_inputs(
        num_inputs=kwargs['miniBatch'],
        input_size=kwargs['inputSize'],
        min_seq_len=kwargs['seqLength'] - 30,
        max_seq_len=kwargs['seqLength'] + 30,
        device=kwargs['device'],
        seed=17)
    params, _ = make_weights(
        input_size=kwargs['inputSize'],
        hidden_size=kwargs['hiddenSize'],
        device=kwargs['device'])

    # Be careful with the indexing... :/
    inputs = [x, hx, cx] + params[0]
    rnn = varlen_lstm_factory(lstm_cell, script)
    return rnn, inputs, flatten_list([params[0]])


def varlen_batching_factory(cell, script):
    # TODO: there are some RCB issues with torch.jit.batch
    # TODO: List seems broken...
    # TODO: doesn't support t() -- does the batch graph support non-batched ops? o.O
    # TODO: support (batched, non-batched) ops (torch.mm(batched, weight))
    #       in auto batching pass
    # TODO: support chunk()
    @torch.jit.batch(batch_size=64)
    def dynamic_rnn(input, hx, cx, wih, whh, bih, bhh):
        # output = []
        hy = hx
        cy = cx
        for seq_idx in range(input.size(0)):
            x = input.select(0, seq_idx)
            gates = torch.matmul(x, wih) + torch.matmul(hx, whh) + bih + bhh

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            # output += [hy]

        #return torch.stack(output), hy.unsqueeze(0), cy.unsqueeze(0)
        return hy.unsqueeze(0), cy.unsqueeze(0)

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def make_varlen_batching_inputs(num_inputs=64, input_size=512, hidden_size=512,
                                min_seq_len=60, max_seq_len=120, device='cuda', seed=17):
    torch.manual_seed(seed)
    seq_lens = torch.randint(min_seq_len, max_seq_len, [num_inputs]).sort(descending=True)[0]
    hx = [torch.randn(1, 1, hidden_size, device=device)
          for i in range(num_inputs)]
    cx = [torch.randn(1, 1, hidden_size, device=device)
          for i in range(num_inputs)]
    x = [torch.randn(1, seq_lens[i], input_size, device=device)
         for i in range(num_inputs)]
    return x, hx, cx


def varlen_batching_creator(script=True, **kwargs):
    x, hx, cx = make_varlen_batching_inputs(
        num_inputs=kwargs['miniBatch'],
        input_size=kwargs['inputSize'],
        min_seq_len=kwargs['seqLength'] - 30,
        max_seq_len=kwargs['seqLength'] + 30,
        device=kwargs['device'],
        seed=17)
    params, _ = make_weights(
        input_size=kwargs['inputSize'],
        hidden_size=kwargs['hiddenSize'],
        device=kwargs['device'])

    # Be careful with the indexing... :/
    x_batched = torch.jit.BatchTensor(x, torch.tensor([1, 0]).byte())
    hx_batched = torch.jit.BatchTensor(hx, torch.tensor([0, 0]).byte())
    cx_batched = torch.jit.BatchTensor(cx, torch.tensor([0, 0]).byte())

    inputs = [x_batched, hx_batched, cx_batched] + params[0]
    inputs[3] = inputs[3].data.t_()
    inputs[4] = inputs[4].data.t_()

    rnn = varlen_batching_factory(lstm_cell, script)
    return rnn, inputs, flatten_list([params[0]])
