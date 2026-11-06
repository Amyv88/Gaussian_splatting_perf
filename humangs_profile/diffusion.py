from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int):
        super().__init__()
        groups_in = 8 if in_channels % 8 == 0 else 1
        groups_out = 8 if out_channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.embed = nn.Linear(embedding_dim, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.embed(F.silu(embedding))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class Denoiser(nn.Module):
    """Epsilon-prediction UNet. Conditioning is added to the time embedding."""

    def __init__(self, cond_dim: int = 12, base: int = 32):
        super().__init__()
        embedding_dim = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.input = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base, embedding_dim)
        self.down_sample1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock(base * 2, base * 2, embedding_dim)
        self.down_sample2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.mid = ResBlock(base * 4, base * 4, embedding_dim)
        self.up_sample2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2 = ResBlock(base * 4, base * 2, embedding_dim)
        self.up_sample1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up1 = ResBlock(base * 2, base, embedding_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))
        self.base = base

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        embedding = self.time_mlp(timestep_embedding(timesteps, self.base)) + self.cond_mlp(cond)
        h1 = self.down1(self.input(x), embedding)
        h2 = self.down2(self.down_sample1(h1), embedding)
        hidden = self.mid(self.down_sample2(h2), embedding)
        hidden = self.up_sample2(hidden)
        hidden = self.up2(torch.cat([hidden, h2], dim=1), embedding)
        hidden = self.up_sample1(hidden)
        hidden = self.up1(torch.cat([hidden, h1], dim=1), embedding)
        return self.output(hidden)


def make_schedule(steps: int, device) -> dict[str, torch.Tensor]:
    """Ho et al. linear betas, scaled so any step count matches the 1000-step curve."""
    scale = 1000.0 / steps
    beta = torch.linspace(1e-4 * scale, 0.02 * scale, steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return {"beta": beta, "alpha": alpha, "alpha_bar": alpha_bar, "steps": steps}


def condition_vector(yaw: torch.Tensor, elev: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """yaw, elev: (B,). pose: (B, 8) or (8,) broadcast."""
    if pose.ndim == 1:
        pose = pose[None].expand(yaw.shape[0], -1)
    return torch.cat(
        [yaw.sin()[:, None], yaw.cos()[:, None], elev.sin()[:, None], elev.cos()[:, None], pose],
        dim=1,
    )


def q_sample(x0: torch.Tensor, timesteps: torch.Tensor, epsilon: torch.Tensor, alpha_bar: torch.Tensor):
    bar = alpha_bar[timesteps][:, None, None, None]
    return bar.sqrt() * x0 + (1.0 - bar).sqrt() * epsilon


class SpecifyGradient(torch.autograd.Function):
    """Backward returns `grad` and does not flow into it."""

    @staticmethod
    def forward(ctx, x, grad):
        ctx.save_for_backward(grad.detach())
        return x.new_ones(())

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad * grad_output, None


def sds_surrogate(x0, timesteps, epsilon_hat, epsilon, alpha_bar):
    """Inject w * (eps_hat - eps) as dL/dx0. Mean reduction would shrink it by the pixel count."""
    bar = alpha_bar[timesteps][:, None, None, None]
    weight = (1.0 - bar).clamp(min=0.05)
    residual = weight * (epsilon_hat - epsilon)
    return SpecifyGradient.apply(x0, residual)
