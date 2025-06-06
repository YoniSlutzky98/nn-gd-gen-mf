import torch

from torch.optim.optimizer import Optimizer


class GroupRMSprop(Optimizer):
    """
    Adaptive learning rate optimizer, similar in essence to RMSprop. Divides learning rate by the square root of an exponentially weighted
    average of squared gradient norms.
    Adapted from appendix D.2 of https://arxiv.org/abs/2201.11729.
    """

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 alpha: float = 0.99,
                 eps: float = 1e-6,
                 weight_decay: float = 0,
                 momentum: float = 0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, adjusted_lr=lr)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                device = next(iter(group['params'])).device
                dtype = next(iter(group['params'])).dtype
                state['square_avg'] = torch.tensor(0., device=device, dtype=dtype)
                state['velocity'] = {}

            square_avg = state['square_avg']
            alpha = group['alpha']
            square_avg.mul_(alpha)

            state['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                if grad.is_sparse:
                    raise RuntimeError('GroupRMSprop does not support sparse gradients')

                square_avg.add_((1 - alpha) * grad.pow(2).sum())

            avg = square_avg.div(1 - alpha ** state['step']).sqrt_().add_(group['eps'])
            lr = group['lr'] / avg
            group['adjusted_lr'] = lr

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                if momentum > 0.:
                    buf = state['velocity'].get(p)
                    if buf is None:
                        buf = state['velocity'][p] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad

                p.data.add_(-lr.to(grad.device) * grad)

        return loss