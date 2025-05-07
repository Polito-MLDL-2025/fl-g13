from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


# Reference:    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
#               https://github.com/pytorch/pytorch/blob/v2.7.0/torch/optim/sgd.py#L26

class SparseSGDM(Optimizer):
    def __init__(self, params, mask: List[Tensor], lr: float, momentum: float = 0.0, dampening: float = 0.0,
                 weight_decay: float = 0.00, nesterov: bool = False, maximize=False) -> None:

        # Checks
        if not isinstance(mask, list):
            raise ValueError("Mask must be a list of tensors.")
        for m in mask:
            if not isinstance(m, Tensor):
                raise ValueError("Each mask element must be a torch.Tensor.")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # Prepare defaults values
        defaults = dict(mask=mask, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, maximize=maximize)
        super().__init__(params, defaults)  # params are moved to self.param_groups

        # Group-wise parameters not supported at the moment, passing them for torch compatibility
        for param_group in self.param_groups:
            param_group['mask'] = mask
            param_group['lr'] = lr
            param_group['momentum'] = momentum
            param_group['dampening'] = dampening
            param_group['weight_decay'] = weight_decay
            param_group['nesterov'] = nesterov
            param_group['maximize'] = maximize

        # Sanity-check
        for param_group in self.param_groups:
            for p, m in zip(param_group['params'], param_group['mask']):
                if p.shape != m.shape:
                    raise ValueError("Mask shape must match the shape of the corresponding parameter.")

    def set_mask(self, mask):
        # Checks
        if not isinstance(mask, list):
            raise ValueError("Mask must be a list of tensors.")
        for m in mask:
            if not isinstance(m, Tensor):
                raise ValueError("Each mask element must be a torch.Tensor.")
        for param_group in self.param_groups:
            param_group['mask'] = mask

        # Sanity-check
        for param_group in self.param_groups:
            for p, m in zip(param_group['params'], param_group['mask']):
                if p.shape != m.shape:
                    raise ValueError("Mask shape must match the shape of the corresponding parameter.")
        self.defaults['mask'] = mask

    # TODO?: sparsity management for mask
    # TODO? dynamic require_grad control: set to True or False and completely skip them in the Differentiation Graph
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            masks           = param_group['mask']
            lr              = param_group['lr']
            momentum        = param_group['momentum']
            dampening       = param_group['dampening']
            weight_decay    = param_group['weight_decay']
            nesterov        = param_group['nesterov']
            maximize        = param_group['maximize']

            for p, m in zip(param_group['params'], masks):
                if p.grad is None:
                    continue

                # Detached for safety, not cloned as done in:
                # https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/optim/sgd.py#L93)
                grad = p.grad.detach()  # .clone()
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)  # Modify tensor in-place using add_

                if momentum != 0:
                    state = self.state[p]
                    buf = state.get('momentum_buffer', None)

                    if buf is None:
                        # Cloning the grad, but in a faster way that as done in:
                        # https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/optim/sgd.py#L351
                        buf = p.grad.detach().clone()
                        state['momentum_buffer'] = buf
                    else:
                        buf.mul_(momentum).add_(grad, alpha=(1 - dampening))

                    if nesterov:
                        # Could use add_ to optimize operations, but nesterov is rarely used, preferred following as done in:
                        # https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/optim/sgd.py#L357
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        # Not cloned, suspicious, but this is how it is done in:
                        # https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/optim/sgd.py#L359
                        grad = buf

                if maximize:
                    grad.neg_()

                # Uncomment this in the future if we are sure we do not need gradients after multiplication is done,
                # while also substituting grad*m with grad in lines below
                # 
                # Apply mask to gradient directly (safest + most common approach)
                # grad.mul_(m) # If m is zero, the grad is completely lost!

                # Apply mask and step
                p.data.add_(grad * m, alpha=-lr)

        return loss
