# https://github.com/IST-DASLab/sparsegpt/blob/master/sparsegpt.py

import math
import time

import torch
import torch.nn as nn
import transformers

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def hessian(inp, baseline=False):
    nsamples = inp.shape[0]
    if nsamples == 0 or baseline:
        # Simulate RTN by returning and identity Hessian
        return torch.eye(inp.shape[-1], device=inp.device)
    inp = inp.float()
    inp = inp.reshape((-1, inp.shape[-1]))
    H = inp.t().matmul(inp)
    H /= 2 / nsamples
    return H


# Adapted from https://github.com/IST-DASLab/qmoe

def batch_sparsegpt(
        W, H, sparsity, blocksize=128, percdamp=.1
):
    dtype = W.dtype
    W = W.clone()
    W = W.float()

    rows, columns = W.shape[1:]
    dev = W.device

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    diag = torch.arange(columns, device=dev)
    damp = percdamp * torch.mean(H[:, diag, diag], axis=-1, keepdim=True)
    damp = torch.maximum(damp, 1e-6 * torch.ones_like(damp))  # catch all zeros
    H[:, diag, diag] += damp

    err = True
    while err:
        # We need to loop as batch operations only return the first error
        try:
            H1 = torch.linalg.cholesky(H)
            H1 = torch.cholesky_inverse(H1)
            H1 = torch.linalg.cholesky(H1, upper=True)
            H = H1
            err = False
        except RuntimeError as ex:
            print('Skip due to singularity.')
            idx = int(str(ex).replace('linalg.cholesky: (Batch element ', '').split('):')[0])
            # Do RTN for failed Hessians by turning them into identity
            H[idx] = torch.eye(columns, device=dev)
    Hinv = H

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, :, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]

        diag = torch.arange(count, device=dev)
        tmp = W1 ** 2 / Hinv1[:, diag, diag].unsqueeze(1) ** 2
        thresh = torch.sort(tmp.flatten(1), 1)[0][:, int(tmp[0].numel() * sparsity)]
        mask1 = tmp <= thresh.reshape((-1, 1, 1))

        for i in range(count):
            w = W1[:, :, i]
            d = Hinv1[:, i, i].unsqueeze(1)
            q = w.clone()
            q[mask1[:, :, i]] = 0
            Q1[:, :, i] = q
            Losses1[:, :, i] = (w - q) ** 2 / d ** 2
            err1 = (w - q) / d
            W1[:, :, i:] -= torch.bmm(err1.unsqueeze(2), Hinv1[:, i, i:].unsqueeze(1))
            Err1[:, :, i] = err1

        Q[:, :, i1:i2] = Q1
        Losses[:, :, i1:i2] = Losses1 / 2

        W[:, :, i2:] -= torch.bmm(Err1, Hinv[:, i1:i2, i2:])

    torch.cuda.synchronize(device=dev)
    print('error', torch.sum(Losses.flatten(1), 1))
    print('Sparsity:', torch.mean((Q == 0).float()))

    return Q.to(dtype)


class SparseGPTSuperFast:

    def __init__(self, layer, model_num):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.nsamples = 0
        self.model_counter = 0

        if isinstance(self.layer, nn.Conv2d) and self.layer.groups > 1:
            self.H = torch.zeros((model_num, W.shape[0], self.columns, self.columns), device=self.dev)
        else:
            self.H = torch.zeros((model_num, self.columns, self.columns), device=self.dev)

    def inc_model_counter(self):
        self.model_counter += 1
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) >= 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            channels = inp.shape[1]
            inp = unfold(inp)

            if self.layer.groups == 1:
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
            else:
                inp = inp.reshape((inp.shape[0], channels, inp.shape[1] // channels, inp.shape[2]))
                inp = inp.permute([2, 0, 1, 3])

        # 
        if isinstance(self.layer, nn.Conv2d) and self.layer.groups > 1:
            inp = inp.flatten(2)

        self.H[self.model_counter] *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H[self.model_counter] += inp.matmul(inp.t())

    def fasterprune(
            self, sparsity, layers, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        assert len(layers) == self.H.shape[0]
        #         print("fast prune, sparsity:", sparsity)
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        W = W.unsqueeze(0)
        repeat_tuple = [1] * len(W.shape)
        repeat_tuple[0] = len(layers)
        Qs = batch_sparsegpt(W.repeat(*repeat_tuple), self.H, sparsity, blocksize=blocksize, percdamp=percdamp)
        if isinstance(self.layer, transformers.Conv1D):
            Qs = Qs.t()
        for i, layer in enumerate(layers):
            layer.weight.data = Qs[i].reshape(layer.weight.shape).to(layer.weight.data.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
