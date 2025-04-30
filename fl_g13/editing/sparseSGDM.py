import torch
from torch.optim.optimizer import Optimizer

class SparseSGDM(Optimizer):
    def __init__(self, params, mask, lr, maximize=False):

        if not isinstance(mask, list):
            raise ValueError("Mask must be a list of tensors.")

        for m in mask:
            if not isinstance(m, torch.Tensor):
                raise ValueError("Each mask element must be a torch.Tensor.")

        defaults = dict(lr=lr, maximize=maximize)
        super().__init__(params, defaults) # params are moved to self.param_groups

        for param_group in self.param_groups:
            param_group['lr'] = lr  
            param_group['mask'] = mask

        for param_group in self.param_groups:
            for p, m in zip(param_group['params'], param_group['mask']):
                if p.shape != m.shape:
                    raise ValueError("Mask shape must match the shape of the corresponding parameter.")


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        maximize = self.defaults['maximize']
        for param_group in self.param_groups:
            lr = param_group['lr']
            masks = param_group['mask']
            for p, m in zip(param_group['params'], masks):
                if p.grad is None:
                    continue
                grad = p.grad
                if maximize:
                    grad = -grad
                p.data -= lr * grad * m

        return loss

