import torch
import torch.nn as nn
import torch.nn.functional as F
from .discretize import discretize_simple
from .constants import D_MODEL, D_STATE, D_INNER, D_CONV

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, d_state=D_STATE, d_inner=D_INNER, d_conv=D_CONV, dt_rank=48):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        
        self.norm = RMSNorm(d_model)
        
        # Projections
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # Conv
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False) # dt, B, C
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        
        # A parameter (constant)
        # In official Mamba, A is learned as log(A)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).expand(d_inner, d_state).float()))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Inference state
        self.register_buffer("conv_state", torch.zeros(1, d_inner, d_conv))
        self.register_buffer("ssm_state", torch.zeros(1, d_inner, d_state))

    def step(self, x_t):
        """
        Single token inference step.
        x_t: (batch, d_model)
        """
        x_t = self.norm(x_t)
        
        # 1. Projections
        xz = self.in_proj(x_t) # (batch, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1) # (batch, d_inner)
        
        # 2. Conv step (manual shift)
        # self.conv_state: (batch, d_inner, d_conv)
        self.conv_state = torch.cat([self.conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        # Apply conv
        # Depthwise conv: (batch, d_inner, d_conv) * (d_inner, d_conv) -> (batch, d_inner)
        x_conv = torch.sum(self.conv_state * self.conv1d.weight.squeeze(1), dim=-1) + self.conv1d.bias
        x_conv = F.silu(x_conv)
        
        # 3. SSM step
        # Project x_conv to dt, B, C
        dt_bc = self.x_proj(x_conv) # (batch, dt_rank + 2 * d_state)
        dt, B, C = torch.split(dt_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Discretize
        # dt_proj(dt) gives delta
        delta = F.softplus(self.dt_proj(dt)) # (batch, d_inner)
        
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Simple discretization (could be more complex, but this shows the point)
        A_bar, B_bar = discretize_simple(delta, A, B) # (batch, d_inner, d_state)
        
        # h_t = A_bar * h_{t-1} + B_bar * x_t
        # Note: x_t here is x_conv
        self.ssm_state = A_bar * self.ssm_state + B_bar * x_conv.unsqueeze(-1)
        
        # y_t = C * h_t + D * x_t
        # C is (batch, d_state), h_t is (batch, d_inner, d_state)
        # We need (batch, d_inner)
        y_ssm = torch.sum(self.ssm_state * C.unsqueeze(1), dim=-1)
        y_t = y_ssm + self.D * x_conv
        
        # 4. Gating and output projection
        y_t = y_t * F.silu(z)
        out = self.out_proj(y_t)
        
        return out

class MambaRNN(nn.Module):
    def __init__(self, n_layers, d_model=D_MODEL, dt_rank=48):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model=d_model, dt_rank=dt_rank) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        
    def step(self, x_t):
        for layer in self.layers:
            # Residual connection
            x_t = x_t + layer.step(x_t)
        return self.norm_f(x_t)

    def reset_state(self):
        for layer in self.layers:
            layer.conv_state.zero_()
            layer.ssm_state.zero_()

class MambaLM(nn.Module):
    def __init__(self, n_layers, d_model=D_MODEL, vocab_size=50277, dt_rank=48): # GPT-NeoX vocab size
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = MambaRNN(n_layers, d_model, dt_rank=dt_rank)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight # Weight tying
        
    def step(self, token_id):
        """
        token_id: (batch,) integer
        Returns: logits (batch, vocab_size)
        """
        x_t = self.embedding(token_id) # (batch, d_model)
        hidden = self.backbone.step(x_t)
        logits = self.lm_head(hidden)
        return logits

    def reset_state(self):
        self.backbone.reset_state()
