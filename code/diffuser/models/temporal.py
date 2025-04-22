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
        num_embeddings=100,
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

        self.history_encoder = ConvLSTMModel(hidden_dim=dim, output_dim=obs_cond_dim)
        # self.history_encoder2 = SpatiotemporalTransformer(in_channels=obs_cond_dim, embed_dim=dim, num_heads=4)
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=8)

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

    def forward(self, x, cond, time, dummy_cond=None, cond_obs=None, cond_mask=None, cond_im=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ] # full trajectory where first state matches cond_obs
            cond: [ batch x cond_dim ] # text embedding
            dummy_cond: [ batch x cond_dim ] # empty string
            cond_obs: [ batch x transition ] # first state
            cond_im: [ batch x C x H x W ] # first state
        '''
        print("x.shape", x.shape, x.dtype, "cond.shape", cond.shape, cond.dtype, "dummy_cond.shape", dummy_cond.shape, dummy_cond.dtype,
              "cond_obs.shape", cond_obs.shape, cond_obs.dtype, "time.shape", time.shape)
        
        x = einops.rearrange(x, 'b h t -> b t h')
        # import pdb; pdb.set_trace()

        t = self.time_mlp(time)
        cond_obs_encoded = self.history_encoder(cond_obs)
        # cond_obs_encoded2 = self.history_encoder2(cond_obs, cond_mask)
        
        input_cond = cond
        # print(input_cond.shape, dummy_cond.shape, cond_obs_encoded.shape)
        if self.returns_condition:
            assert dummy_cond is not None
            if use_dropout:
                if (self.mask_dist.sample(sample_shape=(dummy_cond.size(0), 1)).detach().cpu().numpy().flatten()[0] == 0.0): #10% replace with fake cond 
                    input_cond = dummy_cond
            if force_dropout:
                input_cond = dummy_cond #replace with fake cond
        if cond_im is not None:
            cond_im = torch.cat([cond_obs, self.resnet18(cond_im).squeeze(2,3)], dim=-1)
        
        input_cond = self.embedding(input_cond.long())
        # print(input_cond.shape, dummy_cond.shape, cond_obs_encoded.shape)   
        # import pdb; pdb.set_trace()
        print(t.shape, input_cond.shape, cond_obs_encoded.shape)
        t = torch.cat([t, input_cond, cond_obs_encoded], dim=-1)
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


class ConvLSTMModel(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=10):
        super(ConvLSTMModel, self).__init__()
        
        # Conv2D encoder to process spatial input [C, W, H]
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels=26, out_channels=32, kernel_size=3, padding=1),  # [B*T, 32, 8, 5]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                            # [B*T, 64, 8, 5]
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),                                           # [B*T, 64, 1, 1]
            nn.Flatten(),                                                           # [B*T, 64]
        )
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        B, T, W, H, C = x.shape  # [batch, time, width, height, channel]
        
        # Reshape to [B*T, C, W, H] for Conv2d
        x = x.permute(0, 1, 4, 2, 3).contiguous()      # [B, T, C, W, H]
        x = x.view(B * T, C, W, H)                     # [B*T, C, W, H]

        # Spatial encoder (Conv2d layers)
        x = self.spatial_encoder(x)                   # [B*T, 64]

        # Reshape back to [B, T, 64] for LSTM
        x = x.view(B, T, -1)

        # LSTM
        lstm_out, (hn, _) = self.lstm(x)
        output = self.fc(hn[-1])                      # Use last hidden state
        return output


class SpatiotemporalTransformer(nn.Module):
    def __init__(self, in_channels=26, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(32, 8, 5, embed_dim))  # [T, H, W, D]
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, cond_inputs, cond_masks):
        """
        cond_inputs: [B, T, H, W, C]
        cond_masks: [B, T], 1 = real, 0 = pad
        """
        B, T, H, W, C = cond_inputs.shape
        x = self.embed(cond_inputs)  # [B, T, H, W, D]

        # Add positional encoding (broadcasted across batch)
        x = x + self.pos_embed[:T, :H, :W]  # [B, T, H, W, D]

        # Flatten T x H x W into sequence
        x = x.view(B, T * H * W, -1)  # [B, L, D], L = T*H*W

        # Build attention mask (mask invalid time steps across the whole spatial patch)
        # cond_masks: [B, T] → [B, T, 1, 1] → broadcast to [B, T, H, W]
        mask = cond_masks[:, :, None, None].expand(B, T, H, W).reshape(B, T * H * W)  # [B, L]
        attn_mask = ~mask.bool()  # True = ignore (padded)

        # Multi-head attention expects [B, L, D]
        x_attn, _ = self.attn(x, x, x, key_padding_mask=attn_mask)  # [B, L, D]
        x_out = self.output_proj(x_attn)  # optional projection

        return x_out.view(B, T, H, W, -1)  # return to original 2D+time layout
