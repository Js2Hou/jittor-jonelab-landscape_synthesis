"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/07/05 18:24:28
@Description: 

Methods to be implemented:
- [x] timestep_embedding
- [x] zero_module
- [x] normalization
- [x] conv_nd
- [x] linear
- [x] avg_pool_nd
- [x] timestep_embedding


Classes to be implemented:
- [] GroupNorm32
- [x] SpatialRescaler

"""

import math
import jittor as jt
import jittor.nn as nn
from functools import partial


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = jt.exp(
            -math.log(max_period) * jt.arange(start=0,
                                              end=half, dtype=jt.float32) / half
        )
        # print(type(timesteps), timesteps.shape)
        # print(type(freqs), freqs.shape)
        args = timesteps[:, None].float() * freqs[None]
        embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)
        if dim % 2:
            embedding = jt.concat(
                [embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = timesteps.repeat(dim, 1).transpose()
    return embedding


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    return GroupNorm32(32, channels)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class GroupNorm32(nn.GroupNorm):
    def execute(self, x):
        return super().execute(x.float())


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=2,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=29,
                 out_channels=3,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'bicubic']
        self.multiplier = multiplier
        self.interpolator = partial(jt.nn.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(
                f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(
                in_channels, out_channels, 1, bias=bias)

    def execute(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)
