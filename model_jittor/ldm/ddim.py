"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/07/06 10:58:43
@Description: SAMPLING ONLY.

Classes to be implemented:
- [] DDIMSampler

"""

from typing import Union
import jittor as jt
import numpy as np
from tqdm import tqdm

from utils import toggle_to_eval
from model_jittor.ldm.ddpm import LatentDiffusion
from model_jittor.ldm.utils import (make_ddim_sampling_parameters, 
                                   make_ddim_timesteps, 
                                   noise_like)
                                


class DDIMSampler(object):
    def __init__(self, model: LatentDiffusion, use_ema=False):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        if use_ema:
            self.model.model.load_state_dict(self.model.ema_model.averaged_model.state_dict())
        toggle_to_eval(self.model)

    def register_buffer(self, name, attr):
        if isinstance(attr, jt.Var):
            attr.stop_grad()
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose
        )
        alphas_cumprod = self.model.alphas_cumprod.data
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        def to_var(x: Union[jt.Var, np.ndarray]) -> jt.Var: 
            if isinstance(x, jt.Var): return x.clone()
            elif isinstance(x, np.ndarray): return jt.array(x, jt.float32)

        self.register_buffer('betas', to_var(self.model.betas))
        self.register_buffer('alphas_cumprod', to_var(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_var(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_var(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_var(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_var(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_var(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_var(np.sqrt(1. / alphas_cumprod - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod, ddim_timesteps=self.ddim_timesteps, 
            eta=ddim_eta, verbose=verbose,
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))


    @jt.no_grad()
    def sample(
        self,
        num_steps,
        condition=None,
        quantize_x0=False,
        eta=0.,
        verbose=True,
        log_every_t=100,
    ):
        self.make_schedule(ddim_num_steps=num_steps, ddim_eta=eta, verbose=verbose)
        # sampling
        samples, intermediates = self.ddim_sampling(
            condition, 
            quantize_denoised=quantize_x0, 
            log_every_t=log_every_t
        )
        return samples, intermediates

    @jt.no_grad()
    def ddim_sampling(self, condition, quantize_denoised=False, log_every_t=100):
        img = jt.randn_like(condition)

        timesteps = self.ddim_timesteps
        
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = jt.full((img.shape[0],), step, dtype=jt.int64)

            outs = self.p_sample_ddim(img, condition, ts, index, quantize_denoised)
            img, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @jt.no_grad()
    def p_sample_ddim(self, x, c, t, index, quantize_denoised=False):
        b = x.shape[0]

        e_t = self.model.model(x, t, c) # unet

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = jt.full((b, 1, 1, 1), alphas[index])
        a_prev = jt.full((b, 1, 1, 1), alphas_prev[index])
        sigma_t = jt.full((b, 1, 1, 1), sigmas[index])
        sqrt_one_minus_at = jt.full( (b, 1, 1, 1), sqrt_one_minus_alphas[index])

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
