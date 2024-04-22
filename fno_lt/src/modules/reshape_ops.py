import torch
import torch.nn as nn

__all__ = [
    'FlattenCustom',
    'FlattenNorm',
]


class FlattenCustom(nn.Flatten):
    def __init__(self, cfg, **kwargs):
        start_dim = kwargs['start_dim'] if 'start_dim' in kwargs else 1
        end_dim = kwargs['end_dim'] if 'end_dim' in kwargs else -1
        super(FlattenCustom, self).__init__(start_dim=start_dim, end_dim=end_dim)


class FlattenNorm(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = nn.functional.normalize(x, dim=1)
        return x

