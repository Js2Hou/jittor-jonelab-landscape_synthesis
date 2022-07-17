"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/07/03 19:25:38
@Description: 

Classed to be Implemented:
- [x] VQModel
- [x] VQModelInference

"""
import os

import jittor as jt
import jittor.nn as nn

from .model import Encoder, Decoder
from .quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(nn.Module):
    def __init__(
        self,
        ddconfig,
        n_embed=3,
        embed_dim=8912,
        ckpt_path=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(ddconfig.z_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig.z_channels, 1)

        if ckpt_path is not None:
            assert os.path.isfile(ckpt_path), f"Cannot find ckpt: {ckpt_path}"
            ckpt = jt.load(ckpt_path)
            if isinstance(ckpt, dict) and 'model' in ckpt.keys():
                self.load_state_dict(ckpt['model'])
            else:
                self.load_state_dict(ckpt)
            print(f"Loaded VQModel's ckeckpoint from '{ckpt_path}'")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def execute(self, input, return_pred_indices=False):
        quant, diff, perplexity = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, perplexity

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class VQModelInference(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @jt.no_grad()
    def encode(self, x, force_not_quantize=False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if not force_not_quantize:
            quant, _, _ = self.quantize(h)
        else:
            quant = h
        return quant

    @jt.no_grad()
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    @jt.no_grad()
    def execute(self, input, force_not_quantize=False):
        quant = self.encode(input, force_not_quantize)
        dec = self.decode(quant)
        return dec
