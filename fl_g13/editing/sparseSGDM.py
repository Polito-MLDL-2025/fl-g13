import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

class SparseSGDM(Optimizer):
    def __init__(self, params, mask, lr): # TODO: set types

        # TODO: other checks (size of mask match the one of parameters, shape of the mask, ...)

        defaults = dict(lr=lr)
        super().__init__(params, defaults) # params are moved to self.param_groups

        for i, group in enumerate(self.param_groups):
            group['lr']   = defaults['lr']  
            group['mask'] = mask


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr    = group['lr']
            #print(lr)
            masks = group['mask']
            for p, m in zip(group['params'], masks):
                if p.grad is None:
                    continue
                p.data -= lr * p.grad * m

        return loss

