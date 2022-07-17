"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/07/04 14:17:31
@Description: 

Classes to be Implemented:
- [x] VGGPerceptualLoss
- [x] VQPecptualWithDiscriminator
"""

from collections import namedtuple
import functools
import jittor as jt
import jittor.nn as nn
import jittor.models as models

from torchvision.models import vgg16 as tvgg16

from typing import Tuple

class BaseCriterion(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseCriterion, self).__init__(*args, **kwargs)
        self.eval()
        
    def execute(self, 
                output: jt.Var,
                target: jt.Var, 
                generator: nn.Module, 
                discriminator: nn.Module) -> Tuple[jt.Var, dict]:
        raise NotImplementedError()

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg16 = VGG16()

        self.mean =  jt.array([-.030, -.088, -.188]).view(1, 3, 1, 1).stop_grad()
        self.std = jt.array([.458, .448, .450]).view(1, 3, 1, 1).stop_grad()

    def execute(self, input, target):
        assert input.shape == target.shape

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        # loss = 0.0
        feats_input = self.vgg16(input)
        feats_target = self.vgg16(target)
        # for (feat_input, feat_target) in zip(out_input, out_target):
        #     loss += nn.l1_loss(feat_input, feat_target) 
        # return loss / len(out_input) 
        return nn.l1_loss(feats_input[-1], feats_target[-1])


class VGG16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_jittor = models.vgg16(pretrained=pretrained)
        vgg_pretrained_features = vgg_jittor.features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            self.eval() # jittor apply stop_grad to all param

    def execute(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3


def hinge_d_loss(logits_real, logits_fake):
    loss_real = jt.mean(nn.relu(1. - logits_real))
    loss_fake = jt.mean(nn.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (jt.mean(nn.softplus(-logits_real)) + jt.mean(nn.softplus(logits_fake)))
    return d_loss


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, **kwargs):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        
        self.init_weights()

    
    def init_weights(self):
        def _init_weights(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.gauss_(m.weight, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.gauss_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)


    def execute(self, input):
        return self.main(input)


class AECriterion(BaseCriterion):
    def __init__(self,
                 perceptual_weight: float,
                 gen_weight: float,
                 *args, **kwargs):
        super(AECriterion, self).__init__()
        
        self.perceptual_weight = perceptual_weight
        self.gen_weight = gen_weight
        
        self.perceptual_loss = VGGPerceptualLoss()

    def _calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = jt.grad(nll_loss, last_layer, retain_graph=True)
        g_grads = jt.grad(g_loss, last_layer, retain_graph=True)

        d_weight = jt.norm(nll_grads, dim=[]) / (jt.norm(g_grads, dim=[]) + 1e-4)
        d_weight = jt.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    
    def execute(self, output, target, generator, discriminator, with_gan_loss=False):
        rec_loss = nn.l1_loss(output, target)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(output, target)
        else:
            p_loss = jt.Var(0.)
            p_loss.stop_grad()

        loss = rec_loss + self.perceptual_weight * p_loss

        logits_fake = discriminator(output) # increase logits_fake
        g_loss = -jt.mean(logits_fake) # decrease negetive logits_fake

        if generator.is_train:
            ada_weight = self._calculate_adaptive_weight(loss, g_loss, last_layer=generator.get_last_layer())
        else:
            ada_weight = jt.Var(1)
            ada_weight.stop_grad()

        if with_gan_loss:
            loss += self.gen_weight * ada_weight * g_loss 
        
        logs = {
            "ae_loss": loss,
            "rec_loss": rec_loss,
            "p_loss": p_loss,
            "g_loss": g_loss,
            "ada_weight": ada_weight,
        }
        
        return loss, logs
        

class VQAECriterion(AECriterion):
    def __init__(self, codebook_weight=0.1, 
                 *args, **kwargs):
        super(VQAECriterion, self).__init__(*args, **kwargs)
        self.codebook_weight = codebook_weight
        
    
    def execute(self, output, target, generator, discriminator, q_loss, with_gan_loss=False):
        rec_loss = nn.l1_loss(output, target)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(output, target)
        else:
            p_loss = jt.Var(0.)
            p_loss.stop_grad()

        loss = rec_loss + self.perceptual_weight * p_loss

        logits_fake = discriminator(output) # increase logits_fake
        g_loss = -jt.mean(logits_fake) # decrease negetive logits_fake

        if generator.is_train:
            ada_weight = self._calculate_adaptive_weight(loss, g_loss, last_layer=generator.get_last_layer())
        else:
            ada_weight = jt.Var(1)
            ada_weight.stop_grad()

        if with_gan_loss:
            loss += self.gen_weight * ada_weight * g_loss 
            
        if self.codebook_weight > 0.:
            loss += self.codebook_weight * q_loss
        
        logs = {
            "ae_loss": loss,
            "rec_loss": rec_loss,
            "p_loss": p_loss,
            "g_loss": g_loss,
            "q_loss": q_loss,
            "ada_weight": ada_weight,
        }
        
        return loss, logs

class DiscriminatorCriterion(BaseCriterion):
    def __init__(self,
                 disc_weight: float,
                 disc_loss: str,
                 *args, **kwargs):
        super(DiscriminatorCriterion, self).__init__()
        self.disc_weight = disc_weight
        
        assert disc_loss in ["hinge", "vanilla"]
        
        self.disc_loss = eval(disc_loss + "_d_loss")
        
    def execute(self, output, target, generator, discriminator, with_gan_loss=False):
        logits_real = discriminator(target.detach())
        logits_fake = discriminator(output.detach())

        disc_weight = self.disc_weight if with_gan_loss else 0.
        d_loss = disc_weight * self.disc_loss(logits_real, logits_fake)
        
        log = {
            "disc_loss": d_loss,
            "logits_real": logits_real.mean(),
            "logits_fake": logits_fake.mean(),
        }
        return d_loss, log
