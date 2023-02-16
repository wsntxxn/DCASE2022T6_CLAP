import math
import torch


def cosine_with_warmup(optimizer,
                       train_iterations,
                       warmup_iterations,
                       num_cycles: float = 0.5,
                       last_epoch: int = -1):
    def lr_lambda(iteration):
        if iteration < warmup_iterations:
            return float(iteration) / float(max(1, warmup_iterations))
        progress = float(iteration - warmup_iterations) / float(max(1, train_iterations - warmup_iterations))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, model_size=512, warmup_iters=100, last_epoch=-1, verbose=False):
        self.model_size = model_size
        self.warmup_iters = warmup_iters
        self.factors = [group["lr"] / (self.model_size ** (-0.5) * self.warmup_iters ** (-0.5)) for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)

    def _get_closed_form_lr(self):
        current_iter = self._step_count
        current_lrs = []
        for factor in self.factors:
            current_lr = factor * self.model_size ** (-0.5) * min(current_iter ** (-0.5), current_iter * self.warmup_iters ** (-1.5))
            current_lrs.append(current_lr)
        return current_lrs

    def get_lr(self):
        return self._get_closed_form_lr()
