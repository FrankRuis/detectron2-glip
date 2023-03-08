from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(
            self,
            optimizer,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            eta_min=0,
            last_epoch=-1,
            patience=5,
            verbose=False,
            mode='max'
    ):

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.eta_min = eta_min

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        super(WarmupReduceLROnPlateau, self).__init__(optimizer, factor=gamma, patience=patience, mode=mode,
                                                      min_lr=eta_min, verbose=verbose)

    def step(self, metrics=None):
        warmup_factor = 1

        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            if self.last_epoch >= self.warmup_iters - 1:
                warmup_factor = 1.0

            warmup_lrs = [
                base_lr
                * warmup_factor
                for base_lr in self.base_lrs
            ]

            for param_group, lr in zip(self.optimizer.param_groups, warmup_lrs):
                param_group['lr'] = lr

            self.last_epoch += 1
        elif metrics:
            super().step(metrics)
