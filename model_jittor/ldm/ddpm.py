"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/07/07 19:06:07
@Description: 

Classes to be Implemented:
- [x] LatentDiffusion
- [x] DiffusionWrapper
"""


import os
from contextlib import contextmanager
from functools import partial

import jittor as jt
import jittor.nn as nn
import numpy as np

from model_jittor.autoencoder.vqgan import VQModelInference
from model_jittor.ldm.ema import EMAModel
from model_jittor.ldm.modules import SpatialRescaler
from model_jittor.ldm.openai_model import UNetModel
from model_jittor.ldm.utils import (default, extract_into_tensor,
                                    make_beta_schedule, noise_like)
from tqdm import tqdm


class LatentDiffusion(nn.Module):
    """main class"""

    def __init__(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=0.0015,
        linear_end=0.0205,
        loss_type='l1',
        log_every_t=100,
        image_size=64,
        channels=3,
        unet_config=None,
        first_stage_config=None,
        cond_stage_config=None,
        use_ema=True,  # TODO
        clip_denoised=True,
        ckpt_path=None,
    ):
        super().__init__()
        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end)
        self.use_ema = use_ema
        self.loss_type = loss_type
        self.log_every_t = log_every_t
        self.image_size = image_size
        self.channels = channels
        self.clip_denoised = clip_denoised

        self.first_stage_model = VQModelInference(**first_stage_config)
        self.first_stage_model.eval()

        self.cond_stage_model = SpatialRescaler(**cond_stage_config)
        self.model = DiffusionWrapper(unet_config)
        if self.use_ema: 
            self.ema_model = EMAModel(self.model)
            self.ema_model.register()

        if ckpt_path is not None:
            assert os.path.isfile(ckpt_path), f"Cannot find ckpt: {ckpt_path}"
            ckpt = jt.load(ckpt_path)
            if isinstance(ckpt, dict) and 'model' in ckpt.keys():
                self.load_state_dict(ckpt['model'])
                print(f"using checkpoint from epoch {ckpt['epoch']}")
            else:
                self.load_state_dict(ckpt)
            print(f"Loaded LatentDiffusion's ckeckpoint from '{ckpt_path}'")

    def register_buffer(self, name, attr):
        if isinstance(attr, jt.Var):
            attr.stop_grad()
        setattr(self, name, attr)

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2):
        # jittor may need stop_grad
        betas = make_beta_schedule(beta_schedule, timesteps,
                                   linear_start, linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(jt.array, dtype=jt.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def step_ema(self):
        if self.use_ema:
            self.ema_model.step()
    
    @contextmanager
    def ema_scope(self, enable=False, context=None):
        if self.use_ema and enable:
            self.ema_model.apply_shadow() 
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema and enable: 
                self.ema_model.restore()
                if context is not None:
                    print(f"{context}: Restored training weights")

    def execute(self, x, c, return_encoded=False, *args, **kwargs):
        x = self.encode_first_stage(x).detach()  # B 3 H W -> B 3 H//4 W//4
        t = jt.randint(0, self.num_timesteps, (x.shape[0],)).long().stop_grad()
        c = self.cond_stage_model(c)  # B 29 H W -> B 3 H//4 W//4
        loss = self.p_losses(x, c, t, *args, **kwargs)
        if return_encoded:
            return loss, x.detach()
        else:
            return loss

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: jt.randn_like(x_start).stop_grad())
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_noisy, t, cond)
        loss_simple = self.loss_fn(model_output, noise)
        return loss_simple

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: jt.randn_like(x_start).stop_grad())
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return nn.l1_loss
        elif self.loss_type == 'l2':
            return nn.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    @jt.no_grad()
    def sample_and_decode(self, cond):
        img_latent = self.sample(cond)
        return self.decode_first_stage(img_latent)

    @jt.no_grad()
    def sample(self, cond, return_intermediates=False,
               quantize_denoised=False, log_every_t=None):
        cond = self.cond_stage_model(cond)  # B 29 H W -> B 3 H//4 W//4
        # shape = (cond.shape[0], self.channels, self.image_size, self.image_size)
        shape = cond.shape

        return self.p_sample_loop(cond, shape, return_intermediates,
                                  quantize_denoised=quantize_denoised,
                                  log_every_t=log_every_t)

    @jt.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      quantize_denoised=False, log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        b = shape[0]
        img = jt.randn(shape).stop_grad()
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            ts = jt.full((b,), i, dtype=jt.int64)
            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if i % log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    @jt.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 quantize_denoised=False, return_x0=False, temperature=1.):
        b, *_, _ = *x.shape, 0
        out = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                   quantize_denoised=quantize_denoised,
                                   return_x0=return_x0)
        if return_x0:
            model_mean, _, model_log_variance, x0 = out
        else:
            model_mean, _, model_log_variance = out

        noise = noise_like(x.shape, repeat_noise) * temperature
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, c, t, clip_denoised, quantize_denoised=False,
                        return_x0=False):
        model_out = self.model(x, t, c)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        if clip_denoised:
            x_recon = jt.clamp(x_recon, -1., 1.)
        if quantize_denoised:
            x_recon, _, _ = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @jt.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)

    @jt.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


class DiffusionWrapper(nn.Module):
    def __init__(self, unet_config):
        super().__init__()
        self.diffusion_model = UNetModel(**unet_config)

    def execute(self, x, t, c_concat):
        xc = jt.concat([x, c_concat], dim=1)
        out = self.diffusion_model(xc, t)

        return out
