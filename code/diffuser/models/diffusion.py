from collections import namedtuple
import numpy as np
import torch
from torch import nn

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t, dummy_cond=None, cond_obs=None, cond_im=None, compose=False, **kwargs):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t, dummy_cond=dummy_cond, cond_obs=cond_obs, cond_im=cond_im, compose=compose,  **kwargs)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=True,
        # condition_guidance_w=None,
        condition_guidance_w=1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, x, cond, t, dummy_cond=None, cond_obs=None, cond_im=None, compose=False, uncond_model=None):
        if self.returns_condition:
            if uncond_model:
                epsilon_uncond = uncond_model.model(x, cond, t, dummy_cond, cond_obs, cond_im, force_dropout=True)
            else:
                epsilon_uncond = self.model(x, cond, t, dummy_cond, cond_obs, cond_im, force_dropout=True)
            if uncond_model:
                epsilon_cond = self.model(x, cond, t, dummy_cond, cond_obs, cond_im, use_dropout=False)
                epsilon = epsilon_uncond + epsilon_cond
            elif not compose:
                # epsilon could be epsilon or x0 itself
                epsilon_cond = self.model(x, cond, t, dummy_cond, cond_obs, cond_im, use_dropout=False)
                epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
            elif isinstance(self.condition_guidance_w, float): # multiple cond composition same weight for all
                sum_epsilon_diff = -epsilon_uncond * len(cond)
                for c in cond:
                    sum_epsilon_diff += self.model(x, c.reshape(1,-1), t, dummy_cond, cond_obs, cond_im, use_dropout=False)
                epsilon = epsilon_uncond + (self.condition_guidance_w * sum_epsilon_diff)
            else: # multiple cond composition learned weights
                epsilon_diffs = []
                for c in cond:
                    epsilon_diffs.append(self.model(x, c.reshape(1,-1), t, dummy_cond, cond_obs, cond_im, use_dropout=False) - epsilon_uncond)
                epsilon_diffs = torch.stack(epsilon_diffs, dim=0).squeeze()
                if self.horizon > 1:
                    epsilon = epsilon_uncond + torch.sum(self.condition_guidance_w.reshape(-1,1,1) * epsilon_diffs, dim=0)          
                else:
                    epsilon = epsilon_uncond + torch.sum(self.condition_guidance_w.reshape(-1,1) * epsilon_diffs, dim=0)          
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)        
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon) #classifier-free guidance

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, dummy_cond=None, verbose=True, return_chain=False, sample_fn=default_sample_fn, history_cond=None, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, dummy_cond, **sample_kwargs)
            apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, dummy_cond=None, horizon=None, history_cond=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
                         dim 1 x batch_size x feat_dim?
        '''
        device = self.betas.device
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        if horizon > 1:
            shape = (batch_size, horizon, self.transition_dim)
        else:
            shape = (batch_size, self.transition_dim)

        return self.p_sample_loop(shape, cond, dummy_cond, history_cond=history_cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, dummy_cond=None, cond_obs=None, cond_im=None, invert_model=False, train_uncond=False, uncond_model=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        if train_uncond:
            x_recon = self.model(x_noisy, cond, t, dummy_cond, cond_obs, cond_im, force_dropout=True)
        elif uncond_model:
            epsilon_uncond = uncond_model.model(x_noisy, cond, t, dummy_cond, cond_obs, cond_im, force_dropout=True)
            epsilon_cond = self.model(x_noisy, cond, t, dummy_cond, cond_obs, cond_im, use_dropout=False)
            epsilon_diff = epsilon_cond - epsilon_uncond
            x_recon = epsilon_uncond + (self.condition_guidance_w * epsilon_diff)
        elif not invert_model:
            x_recon = self.model(x_noisy, cond, t, dummy_cond, cond_obs, cond_im)
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)
        elif cond.shape[0] == 1: #single learned condition
            epsilon_uncond = self.model(x_noisy, cond, t, dummy_cond, cond_obs, cond_im, force_dropout=True)
            epsilon_cond = self.model(x_noisy, cond, t, dummy_cond, cond_obs, cond_im, use_dropout=False)
            epsilon_diff = epsilon_cond - epsilon_uncond
            x_recon = epsilon_uncond + (self.condition_guidance_w * epsilon_diff)
        elif isinstance(self.condition_guidance_w, float): # multiple cond composition same weight for all
            epsilon_uncond = self.model(x_noisy, cond, t, dummy_cond, cond_obs, cond_im, force_dropout=True)
            sum_epsilon_diff = -epsilon_uncond * len(cond)
            for c in cond:
                sum_epsilon_diff += self.model(x_noisy, c.reshape(1,-1), t, dummy_cond, cond_obs, cond_im, use_dropout=False)
            x_recon = epsilon_uncond + (self.condition_guidance_w * sum_epsilon_diff)
        else: # multiple cond composition learned weights
            epsilon_uncond = self.model(x_noisy, cond, t, dummy_cond, cond_obs, cond_im, force_dropout=True)
            epsilon_diffs = []
            for c in cond:
                epsilon_diffs.append(self.model(x_noisy, c.reshape(1,-1), t, dummy_cond, cond_obs, cond_im, use_dropout=False) - epsilon_uncond)
            epsilon_diffs = torch.stack(epsilon_diffs, dim=0).squeeze()
            if self.horizon > 1:
                x_recon = epsilon_uncond + torch.sum(self.condition_guidance_w.reshape(-1,1,1) * epsilon_diffs, dim=0)
            else:
                x_recon = epsilon_uncond + torch.sum(self.condition_guidance_w.reshape(-1,1) * epsilon_diffs, dim=0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon: #noise
            loss, info = self.loss_fn(x_recon, noise)
        else: #x0
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info


    def loss(self, x, cond, dummy_cond=None, cond_obs=None, cond_im=None, invert_model=False, train_uncond=False, uncond_model=None):        
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, dummy_cond, cond_obs, cond_im, invert_model, train_uncond, uncond_model)


    def forward(self, cond, dummy_cond=None, history_cond=None, *args, **kwargs):
        return self.conditional_sample(cond, dummy_cond, history_cond=history_cond, *args, **kwargs)

