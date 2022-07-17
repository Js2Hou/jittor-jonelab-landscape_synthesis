"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/07/03 19:25:52
@Description: 

Classes to be Implemented:
- [x] VectorQuantizer
- [x] VectorQuantizer2
"""

import jittor as jt
import jittor.nn as nn
import numpy as np

from ..modules import Embedding


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_embed, embed_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_embed = n_embed  # denote as N
        self.embed_dim = embed_dim  # denote as C
        self.beta = beta

        self.embedding = Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 /
                                            self.n_embed, 1.0 / self.n_embed)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # b c h w -> b h w c
        z = z.permute(0, 2, 3, 1)
        # b h w c -> (b h w), c
        z_flattened = z.reshape(-1, self.embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 - 2 e * z + e^2
        # dist: (b h w) N
        dist = (
            z_flattened.pow(2).sum(1, keepdim=True)
            - 2 * z_flattened @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )

        # find closest encodings
        min_encoding_indices, _ = jt.argmin(
            dist, dim=1, keepdim=True)  # (b h w) 1
        min_encodings = jt.zeros(
            min_encoding_indices.shape[0], self.n_embed).to(z.device)  # (b h w) N
        min_encodings.scatter_(1, min_encoding_indices, 1)  # (b h w) N

        # get quantized latent vectors: (b h w) N x N c -> (b h w) c -> b h w c
        z_q = jt.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        diff = (
            (z_q - z.detach()).square().mean()
            + self.beta * (z - z_q.detach()).mean()
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity, 表示embedding的利用率
        e_mean = jt.mean(min_encodings, dim=0)
        perplexity = jt.exp(-jt.sum(e_mean * jt.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2)  # b h w c -> b c h w
        return z_q, diff, perplexity


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = Embedding(self.n_e, self.e_dim)
        self.embedding.weight.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", jt.array(np.load(self.remap)))  # todo:
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=jt.randint(0,self.re_embed,size=new[unknown].shape)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back = jt.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def execute(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1)
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = jt.sum(z_flattened ** 2, dim=1, keepdims=True) + \
            jt.sum(self.embedding.weight**2, dim=1) - 2 * \
            jt.linalg.einsum('bd,dn->bn', z_flattened, jt.permute(self.embedding.weight, (1, 0)))

        min_encoding_indices, _ = jt.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * jt.mean((z_q.detach()-z)**2) + \
                   jt.mean((z_q - z.detach()) ** 2)
        else:
            loss = jt.mean((z_q.detach()-z)**2) + self.beta * \
                   jt.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2)  # b, c, h, w

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2)

        return z_q