import torch.optim.lr_scheduler as lr_scheduler


class MyStepLR(lr_scheduler.StepLR):
    """
    Override: https://pytorch.org/docs/1.0.0/_modules/torch/optim/lr_scheduler.html#StepLR
    Reason: we want to fix the learning rate to not get lower than some value: min_lr.

    Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        min_lr (float): The lowest allowed value for the learning rate.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, min_lr=1e-6):
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super(lr_scheduler.StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]
