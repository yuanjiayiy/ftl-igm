import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli


from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        obs_cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        returns_condition=True,
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = time_dim + cond_dim + obs_cond_dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, x, cond, time, dummy_cond=None, cond_obs=None, cond_im=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ] # full trajectory where first state matches cond_obs
            cond: [ batch x cond_dim ] # text embedding
            dummy_cond: [ batch x cond_dim ] # empty string
            cond_obs: [ batch x transition ] # first state
            cond_im: [ batch x C x H x W ] # first state
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        input_cond = cond #concept embedding
        if self.returns_condition:
            assert dummy_cond is not None
            if use_dropout:
                if (self.mask_dist.sample(sample_shape=(dummy_cond.size(0), 1)).detach().cpu().numpy().flatten()[0] == 0.0): #10% replace with fake cond 
                    input_cond = dummy_cond
            if force_dropout:
                input_cond = dummy_cond #replace with fake cond
        if cond_im is not None:
            cond_obs = torch.cat([cond_obs, self.resnet18(cond_im).squeeze(2,3)], dim=-1)
        t = torch.cat([t, input_cond, cond_obs], dim=-1)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            tmp = h
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class MLPnet(nn.Module):
    """shapes experiments"""
      
    def __init__(
        self,
        transition_dim=None,
        cond_dim=None,
        dim=128,
        returns_condition=True,
        condition_dropout=0.1,
        calc_energy=False,
        *args, **kwargs
    ):
        super().__init__()

        if calc_energy:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        self.transition_dim = transition_dim
        # self.action_dim = transition_dim - cond_dim

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            # embed_dim = 2*dim
            embed_dim = dim
        else:
            embed_dim = dim

        self.mlp = nn.Sequential(
                        nn.Linear(embed_dim + cond_dim + transition_dim, 1024),
                        act_fn,
                        nn.Linear(1024, 1024),
                        act_fn,
                        nn.Linear(1024, transition_dim),
                    )

    def forward(self, x, cond, time, dummy_cond=None, cond_obs=None, cond_im=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            dummy_cond : [batch x state]
        '''
        # Assumes horizon = 1
        t = self.time_mlp(time)

        input_cond = cond

        if self.returns_condition: #sample (dropout) and maybe use dummy_cond
            assert dummy_cond is not None
            if use_dropout:
                if (self.mask_dist.sample(sample_shape=(dummy_cond.size(0), 1)).detach().cpu().numpy().flatten()[0] == 0.0): #10% replace with fake cond 
                    input_cond = dummy_cond
            if force_dropout:
                input_cond = dummy_cond #replace with fake cond
        inp = torch.cat([t, input_cond, x], dim=-1)
        out  = self.mlp(inp)

        if self.calc_energy:
            energy = ((out - x) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x, create_graph=True)
            return grad[0]
        else:
            return out


