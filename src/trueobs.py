import math
import time

import torch
import torch.nn as nn
import gc

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEBUG = False


class TrueOBS:

    def __init__(self, layer, log_file, always_damp=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.mask = None
        self.always_damp = always_damp
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            self.kernel_size = (W.shape[2] * W.shape[3])
            W = W.flatten(1)

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # Accumulate in double precision34
        if isinstance(self.layer, nn.Conv2d) and self.layer.groups > 1:
            self.H = torch.zeros((W.shape[0], self.columns, self.columns), device=self.dev, dtype=torch.double)
        else:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.double)
        self.nsamples = 0
        self.log_file = log_file
        log = open(self.log_file, "a")  # append mode
        log.write(f"Start logging Hessian rows={self.rows}, columns={self.columns}\n")
        log.close()

    def add_batch(self, inp, out, resolve=False):
        if self.mask != None and isinstance(self.layer, nn.Linear):
            print("masking:", inp.shape, out.shape, self.mask.shape)
            # masking: torch.Size([15, 256, 768]) torch.Size([15, 256, 3072]) torch.Size([15, 256])
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) >= 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            if self.mask is not None:
                #                 print("out:", out.shape)
                #                 print("MASK:",self.mask.reshape(-1).shape, self.mask.reshape(-1)[:40], self.mask.dtype)
                inp = inp[self.mask.reshape(-1).bool()]
                out = out.reshape((-1, out.shape[-1]))[self.mask.reshape(-1).bool()]
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

        if isinstance(self.layer, nn.Conv2d) and self.layer.groups > 1:
            inp = inp.flatten(2)
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()
            if resolve:
                raise NotImplementedError

        else:
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()
            if resolve:
                if isinstance(self.layer, nn.Linear):
                    out = out.t()
                if isinstance(self.layer, nn.Conv2d):
                    out = out.permute([1, 0, 2, 3])
                    out = out.reshape((out.shape[0], -1))
                if not hasattr(self, 'resolvemat'):
                    self.resolvemat = torch.zeros(
                        (inp.shape[0], self.rows), device=self.dev, dtype=torch.double
                    )
                self.resolvemat += inp.matmul(out.t())

    def set_mask(self, mask):
        self.mask = mask

    def resolve(self):
        H = self.H.float()
        H *= self.nsamples / 2
        dead = torch.diag(H) < 1e-3
        H[dead, dead] = 1
        Hinv = self.invert(H, damp=self.nsamples / 2)
        W = Hinv.matmul(self.resolvemat.float()).t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape)

    def invert(self, H, damp=1):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            # Append-adds at last
            log = open(self.log_file, "a")  # append mode
            log.write(
                f"rows={self.rows}, columns={self.columns}, Hessian not full rank. Hessian rank={torch.linalg.matrix_rank(H)}, Hessian shape={H.shape}\n")
            log.close()
            tmp = damp * torch.eye(self.columns, device=self.dev)
            print('Hessian not full rank. tmp:', tmp.shape, "H:", H.shape)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + tmp))
        return Hinv

    def prepare(self, columnslast=False):
        if columnslast:
            perm = torch.arange(self.columns, device=self.dev)
            if len(self.layer.weight.shape) == 4:
                perm = perm.reshape(list(self.layer.weight.shape)[1:])
                perm = perm.permute([1, 2, 0])
                perm = perm.flatten()
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        H = self.H.float()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        W[:, dead] = 0
        if columnslast:
            H = H[perm, :][:, perm]
            W = W[:, perm]

        if self.always_damp != 0:
            tmp = self.always_damp * (torch.trace(H) / H.shape[-1]) * torch.eye(self.columns, device=self.dev)
            H += tmp

        Hinv = self.invert(H)

        Losses = torch.zeros([self.rows, self.columns + 1], device=self.dev)
        if columnslast:
            return W, H, Hinv, Losses, perm
        return W, H, Hinv, Losses

    def prepare_iter(self, i1, parallel, W, Hinv1):
        i2 = min(i1 + parallel, self.rows)
        count = i2 - i1
        w = W[i1:i2, :]
        Hinv = Hinv1.unsqueeze(0).repeat((count, 1, 1))
        mask = torch.zeros_like(w).bool()
        rangecount = torch.arange(count, device=self.dev)
        idxcount = rangecount + i1
        return i2, count, w, Hinv, mask, rangecount, idxcount

    def prepare_sparse(self, w, mask, Hinv, H):
        start = int(torch.min(torch.sum((w == 0).float(), 1)).item()) + 1
        for i in range(w.shape[0]):
            tmp = w[i] == 0
            H1 = H.clone()
            H1[tmp, :] = 0
            H1[:, tmp] = 0
            H1[tmp, tmp] = 1
            Hinv[i] = self.invert(H1)
            mask[i, torch.nonzero(tmp, as_tuple=True)[0][:(start - 1)]] = True
        return start

    def nmprune(self, n=2, m=4, parallel=32):
        W, H, Hinv1, Losses, perm = self.prepare(columnslast=True)

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1)

            buckets = torch.zeros((count, self.columns // m, 1), device=self.dev)

            tick = time.time()

            for zeros in range(1, self.columns + 1):
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = w ** 2 / diag
                tmp = (buckets >= n).repeat((1, 1, m)).flatten(1)
                scores[mask | tmp] = float('inf')
                j = torch.argmin(scores, 1)
                Losses[i1:i2, zeros] = scores[rangecount, j]
                row = Hinv[rangecount, j, :]
                d = diag[rangecount, j]
                w -= row * (w[rangecount, j] / d).unsqueeze(1)
                mask[rangecount, j] = True
                buckets[rangecount, torch.div(j, m, rounding_mode='floor'), :] += 1
                if zeros == self.columns * n / m:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
            Losses[i1:i2, :] /= 2
            w[mask] = 0
            W[i1:i2, :] = w

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

        print('error', torch.sum(Losses).item())
        W = W[:, torch.argsort(perm)]
        self.layer.weight.data = W.reshape(self.layer.weight.shape)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)

    def prepare_unstr(self, parallel=32):
        W, H, Hinv1, Losses = self.prepare()
        self.Losses = Losses
        self.Traces = []

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1)
            start = self.prepare_sparse(w, mask, Hinv, H)

            Trace = torch.zeros((self.columns + 1, count, self.columns), device=self.dev)
            Trace[0, :, :] = w
            Trace[:start, :, :] = w

            tick = time.time()

            # order = torch.argsort(torch.abs(w), 1)
            for zeros in range(start, self.columns + 1):
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = (w ** 2) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores, 1)
                # j = order[:, zeros - 1]
                self.Losses[i1:i2, zeros] = scores[rangecount, j]
                row = Hinv[rangecount, j, :]
                d = diag[rangecount, j]
                w -= row * (w[rangecount, j] / d).unsqueeze(1)
                mask[rangecount, j] = True
                w[mask] = 0
                Trace[zeros, :, :] = w
                if zeros == self.columns:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
            self.Losses[i1:i2, :] /= 2
            self.Traces.append(Trace.cpu())

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

    def prune_unstr(self, sparsities):
        return self.prune_blocked(sparsities)

    def prune_unstr_balanced(self, rem):
        print("columns", self.columns)
        parallel = self.Traces[0].shape[1]
        W = torch.zeros((self.rows, self.columns))
        for i in range(self.rows):
            W[i, :] = self.Traces[i // parallel][self.columns - rem, i % parallel]
        print(torch.sum(self.Losses[:, :(self.columns - rem + 1)]))
        return W

    def prune_struct(self, sparsities, size=1):
        sparsities = sparsities[:]
        W, H, Hinv, Losses = self.prepare()

        count = self.columns // size
        Losses = torch.zeros(count + 1, device=self.dev)
        mask = torch.zeros(count, device=self.dev).bool()
        rangecount = torch.arange(count, device=self.dev)
        rangecolumns = torch.arange(self.columns, device=self.dev)

        tick = time.time()

        res = []
        if 0 in sparsities:
            res.append(W.clone())
            sparsities = sparsities[1:]
        if size == 1:
            for dropped in range(count + 1):
                diag = torch.diagonal(Hinv)
                scores = torch.sum(W ** 2, 0) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped] = scores[j]
                row = Hinv[j, :]
                d = diag[j]
                W -= ((W[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))
                mask[j] = True
                W[:, mask] = 0
                while dropped == math.ceil(count * sparsities[0]):
                    res.append(W.clone())
                    print('%.4f error' % sparsities[0], torch.sum(Losses).item() / 2)
                    sparsities.pop(0)
                    if DEBUG:
                        tmp = self.layer.weight.data.clone()
                        self.layer.weight.data = res[-1].reshape(self.layer.weight.shape)
                        print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)
                        self.layer.weight.data = tmp
                    if not len(sparsities):
                        break
                if not len(sparsities):
                    break
                row /= torch.sqrt(d)
                Hinv -= row.unsqueeze(1).matmul(row.unsqueeze(0))
        else:
            mask1 = torch.zeros(self.columns, device=self.dev).bool()
            for dropped in range(count + 1):
                blocks = Hinv.reshape(count, size, count, size)
                blocks = blocks[rangecount, :, rangecount, :]
                invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                W1 = W.reshape((self.rows, count, size)).transpose(0, 1)
                lambd = torch.bmm(W1, invblocks)
                scores = torch.sum(lambd * W1, (1, 2))
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped] = scores[j]
                rows = Hinv[(size * j):(size * (j + 1)), :]
                d = invblocks[j]
                W -= lambd[j].matmul(rows)
                mask[j] = True
                mask1[(size * j):(size * (j + 1))] = True
                W[:, mask1] = 0
                while dropped == math.ceil(count * sparsities[0]):
                    res.append(W.clone())
                    print('%.4f error' % sparsities[0], torch.sum(Losses).item() / 2)
                    sparsities.pop(0)
                    if DEBUG:
                        tmp = self.layer.weight.data.clone()
                        self.layer.weight.data = res[-1].reshape(self.layer.weight.shape)
                        print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)
                        self.layer.weight.data = tmp
                    if not len(sparsities):
                        break
                if not len(sparsities):
                    break
                Hinv -= rows.t().matmul(d.matmul(rows))
                Hinv[rangecolumns[mask1], rangecolumns[mask1]] = 1

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))

        return res

    def prepare_blocked(self, size=4, parallel=32):
        W, H, Hinv1, Losses, perm = self.prepare(columnslast=True)

        self.Traces = []
        blockcount = self.columns // size
        self.Losses = torch.zeros((self.rows, blockcount + 1), device=self.dev)
        rangeblockcount = torch.arange(blockcount, device=self.dev)
        rangecolumns = torch.arange(self.columns, device=self.dev)

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, _, rangecount, _ = self.prepare_iter(i1, parallel, W, Hinv1)

            mask = torch.zeros((count, blockcount), device=self.dev).bool()
            mask1 = torch.zeros((count, blockcount, size), device=self.dev).bool()
            Trace = torch.zeros((blockcount + 1, count, self.columns), device=self.dev)
            Trace[0, :, :] = w
            rangeblockunroll = torch.arange(count * blockcount, device=self.dev)
            blockdiagidx = rangeblockcount.repeat(count)
            rangeunroll = torch.arange(self.columns * count, device=self.dev)
            diagidx = rangecolumns.repeat(count)
            paroffset = blockcount * rangecount
            expandrows = torch.arange(size, device=self.dev).unsqueeze(0).repeat(count, 1)
            expandrows += self.columns * rangecount.unsqueeze(1)

            tick = time.time()

            for dropped in range(1, blockcount + 1):
                blocks = Hinv.reshape(count * blockcount, size, blockcount, size)
                blocks = blocks[rangeblockunroll, :, blockdiagidx, :]
                invblocks = self.invert(blocks)
                w1 = w.reshape((count * blockcount, 1, size))
                lambd = torch.bmm(w1, invblocks)
                scores = torch.sum(lambd * w1, (1, 2))
                scores = scores.reshape((count, blockcount))
                scores[mask] = float('inf')
                j = torch.argmin(scores, 1)
                self.Losses[i1:i2, dropped] = scores[rangecount, j]

                tmp = (expandrows + size * j.unsqueeze(1)).flatten()
                rows = Hinv.reshape((-1, self.columns))[tmp]
                rows = rows.reshape((count, size, self.columns))
                tmp = paroffset + j
                d = invblocks[tmp]

                w -= torch.bmm(lambd[tmp], rows).squeeze(1)
                mask[rangecount, j] = True
                mask1[mask] = True
                tmp = mask1.flatten(1)
                w[mask1.flatten(1)] = 0
                Trace[dropped, :, :] = w

                if dropped == self.columns:
                    break
                Hinv -= torch.bmm(rows.transpose(1, 2), torch.bmm(d, rows))
                Hinv = Hinv.reshape((count * self.columns, self.columns))
                tmp = mask1.flatten()
                Hinv[rangeunroll[tmp], diagidx[tmp]] = 1
                Hinv = Hinv.reshape((count, self.columns, self.columns))
            self.Losses[i1:i2, :] /= 2
            Trace = Trace[:, :, torch.argsort(perm)]
            self.Traces.append(Trace.cpu())

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

    def prune_blocked(self, sparsities):
        parallel = self.Traces[0].shape[1]
        blockcount = self.Traces[0].shape[0] - 1
        losses = self.Losses[:, 1:].reshape(-1)
        order = torch.argsort(losses)
        Ws = [torch.zeros((self.rows, self.columns), device=self.dev) for _ in sparsities]
        losses = [0] * len(sparsities)
        for i in range(self.rows):
            if i % parallel == 0:
                Trace = self.Traces[i // parallel].to(self.dev)
            for j, sparsity in enumerate(sparsities):
                count = int(math.ceil(self.rows * blockcount * sparsity))
                perrow = torch.sum(
                    torch.div(order[:count], blockcount, rounding_mode='trunc') == i
                ).item()
                losses[j] += torch.sum(self.Losses[i, :(perrow + 1)]).item()
                Ws[j][i, :] = Trace[perrow, i % parallel, :]
        for sparsity, loss in zip(sparsities, losses):
            print('%.4f error' % sparsity, loss)
            if DEBUG:
                tmp = self.layer.weight.data.clone()
                self.layer.weight.data = Ws[sparsities.index(sparsity)].reshape(self.layer.weight.shape)
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)
                self.layer.weight.data = tmp
        return Ws

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
