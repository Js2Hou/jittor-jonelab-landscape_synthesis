import jittor as jt
from jittor.optim import Optimizer
from jittor.lr_scheduler import CosineAnnealingLR
import math

class LinearWarmupCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, 
                 optimizer: Optimizer, 
                 warmup_epochs: int,
                 T_max,
                 warmup_start_lr: float = 0.0,
                 eta_min=0, 
                 last_epoch=-1,
                 by_step=False,
                 epoch_len=None):
        if by_step:
            assert epoch_len is not None
            T_max *= epoch_len
            warmup_epochs *= epoch_len
        super().__init__(optimizer, T_max, eta_min, last_epoch)
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
    
    def get_lr(self, base_lr, now_lr):
        if self.last_epoch == 0:
            return self.warmup_start_lr
        if self.last_epoch < self.warmup_epochs:
            return now_lr + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
        if self.last_epoch == self.warmup_epochs:
            return base_lr
        if (self.last_epoch - 1 - self.T_max) % (2 * (self.T_max - self.warmup_epochs)) == 0:
            return (now_lr + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / (self.T_max - self.warmup_epochs))) / 2)
        return  ((1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) /
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.T_max - self.warmup_epochs))) *
                (now_lr - self.eta_min) + self.eta_min)
    
    