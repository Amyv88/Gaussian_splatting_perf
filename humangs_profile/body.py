"""Forward-kinematics body. Each Gaussian is parented to one rigid bone."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


POSE_DIM = 8
# 0 left shoulder lift, 1 right shoulder lift, 2 left elbow, 3 right elbow,
# 4 left hip abduction, 5 right hip abduction, 6 left knee, 7 right knee.

SKIN = (0.83, 0.64, 0.50)
HAIR = (0.13, 0.08, 0.06)
EYE = (0.06, 0.06, 0.07)
SHIRT = (0.15, 0.34, 0.74)
PANTS = (0.17, 0.19, 0.26)
SHOE = (0.07, 0.07, 0.08)


def rest_pose(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(POSE_DIM, device=device, dtype=dtype)


def sample_pose(n: int, device=None, generator: torch.Generator | None = None) -> torch.Tensor:
    span = torch.tensor([0.5, 0.5, 0.6, 0.6, 0.25, 0.25, 0.4, 0.4], device=device)
    if generator is None:
        noise = torch.rand(n, POSE_DIM, device=device) * 2 - 1
    else:
        noise = torch.rand(n, POSE_DIM, device=device, generator=generator) * 2 - 1
    return noise * span


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm().clamp(min=eps)


def _rot_z(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    z = torch.zeros((), device=angle.device, dtype=angle.dtype)
    o = torch.ones((), device=angle.device, dtype=angle.dtype)
    return torch.stack(
        [
            torch.stack([c, -s, z]),
            torch.stack([s, c, z]),
            torch.stack([z, z, o]),
        ]
    )


def _frame(direction: torch.Tensor) -> torch.Tensor:
    y_axis = _normalize(direction)
    hint = y_axis.new_tensor([0.0, 1.0, 0.0])
    x_axis = torch.linalg.cross(hint, y_axis)
    if float(x_axis.norm()) < 1e-5:
        hint = y_axis.new_tensor([1.0, 0.0, 0.0])
        x_axis = torch.linalg.cross(hint, y_axis)
    x_axis = _normalize(x_axis)
    z_axis = _normalize(torch.linalg.cross(x_axis, y_axis))
    return torch.stack([x_axis, y_axis, z_axis], dim=1)


def _bend(direction: torch.Tensor, target: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    direction = _normalize(direction)
    target = _normalize(target)
    axis = torch.linalg.cross(direction, target)
    if float(axis.norm()) < 1e-6:
        return direction
    axis = _normalize(axis)
    c, s = torch.cos(angle), torch.sin(angle)
    return (
        direction * c
        + torch.linalg.cross(axis, direction) * s
        + axis * torch.dot(axis, direction) * (1 - c)
    )


def _covariance(rotation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    diag = torch.diag_embed(scale.square())
    return rotation @ diag @ rotation.transpose(1, 2)


class HumanAvatar(nn.Module):
    """`template=True` uses fixed part colors. Otherwise color, opacity, scale, and the mean residual are parameters."""

    def __init__(self, density: float = 2.0):
        super().__init__()
        self.density = float(density)
        pose = rest_pose()
        _means, _rotation, base_scale, template_color = self._build(pose)
        self.register_buffer("base_scale", base_scale)
        self.register_buffer("template_color", template_color)
        n = template_color.shape[0]
        self.color_logit = nn.Parameter(torch.zeros(n, 3))
        # sigmoid(1.4) is about 0.80, so the gray init is already a solid silhouette.
        self.opacity_logit = nn.Parameter(torch.full((n,), 1.4))
        self.mean_residual = nn.Parameter(torch.zeros(n, 3))
        self.log_scale = nn.Parameter(torch.zeros(n, 3))

    @property
    def count(self) -> int:
        return int(self.template_color.shape[0])

    def _n(self, base: int) -> int:
        return max(1, int(round(base * self.density)))

    def gaussians(self, pose: torch.Tensor, template: bool = False):
        means, rotation, _scale, _color = self._build(pose)
        if template:
            scale = self.base_scale
            color = self.template_color
            opacity = means.new_full((self.count,), 0.90)
        else:
            if means.shape[0] != self.count:
                raise RuntimeError(
                    f"pose changed the Gaussian count from {self.count} to {means.shape[0]}"
                )
            scale = self.base_scale * torch.exp(self.log_scale.clamp(-1.2, 1.2))
            means = means + self.mean_residual
            color = torch.sigmoid(self.color_logit)
            opacity = torch.sigmoid(self.opacity_logit)
        return means, _covariance(rotation, scale), color, opacity

    def _build(self, pose: torch.Tensor):
        device, dtype = pose.device, pose.dtype
        means_all, rot_all, scale_all, color_all = [], [], [], []

        def add(origin, direction, length, n, rx, rz, color):
            direction = _normalize(direction)
            rotation = _frame(direction)
            t = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, device=device, dtype=dtype)
            local = torch.zeros(n, 3, device=device, dtype=dtype)
            local[:, 1] = t * length
            means = origin + local @ rotation.transpose(0, 1)
            scale = torch.zeros(n, 3, device=device, dtype=dtype)
            scale[:, 0] = rx
            scale[:, 1] = (length / n) * 0.72
            scale[:, 2] = rz
            means_all.append(means)
            rot_all.append(rotation.expand(n, 3, 3).clone())
            scale_all.append(scale)
            color_all.append(means.new_tensor(color).expand(n, 3).clone())

        def add_cloud(means, scale_value, color):
            n = means.shape[0]
            eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3).clone()
            scale = torch.full((n, 3), float(scale_value), device=device, dtype=dtype)
            means_all.append(means)
            rot_all.append(eye)
            scale_all.append(scale)
            color_all.append(means.new_tensor(color).expand(n, 3).clone())

        pose = pose.to(dtype=dtype)
        shoulder_l = pose.new_tensor([-0.20, 1.36, 0.0])
        shoulder_r = pose.new_tensor([0.20, 1.36, 0.0])
        left_dir = _normalize(
            _rot_z(pose.new_tensor(0.85) - pose[0]) @ pose.new_tensor([-1.0, 0.0, 0.0])
        )
        right_dir = _normalize(
            _rot_z(pose.new_tensor(-0.85) + pose[1]) @ pose.new_tensor([1.0, 0.0, 0.0])
        )
        upper_len, fore_len = 0.30, 0.28
        elbow_l = shoulder_l + left_dir * upper_len
        elbow_r = shoulder_r + right_dir * upper_len
        fore_l = _bend(left_dir, pose.new_tensor([0.0, -0.2, 1.0]), pose[2] + 0.25)
        fore_r = _bend(right_dir, pose.new_tensor([0.0, -0.2, 1.0]), pose[3] + 0.25)
        hip_l = pose.new_tensor([-0.11, 0.92, 0.0])
        hip_r = pose.new_tensor([0.11, 0.92, 0.0])
        thigh_l = _normalize(_rot_z(-pose[4]) @ pose.new_tensor([0.0, -1.0, 0.0]))
        thigh_r = _normalize(_rot_z(pose[5]) @ pose.new_tensor([0.0, -1.0, 0.0]))
        thigh_len, shin_len = 0.42, 0.40
        knee_l = hip_l + thigh_l * thigh_len
        knee_r = hip_r + thigh_r * thigh_len
        shin_l = _bend(thigh_l, pose.new_tensor([0.0, 0.0, -1.0]), pose[6] + 0.12)
        shin_r = _bend(thigh_r, pose.new_tensor([0.0, 0.0, -1.0]), pose[7] + 0.12)

        add(pose.new_tensor([0.0, 0.98, 0.0]), pose.new_tensor([0.0, 1.0, 0.0]), 0.40, self._n(8), 0.17, 0.11, SHIRT)
        add(pose.new_tensor([0.0, 0.90, 0.0]), pose.new_tensor([0.0, 1.0, 0.0]), 0.12, self._n(3), 0.16, 0.11, PANTS)
        add(pose.new_tensor([0.0, 1.38, 0.0]), pose.new_tensor([0.0, 1.0, 0.0]), 0.10, self._n(2), 0.055, 0.055, SKIN)
        add(shoulder_l, left_dir, upper_len, self._n(5), 0.050, 0.050, SHIRT)
        add(elbow_l, fore_l, fore_len, self._n(5), 0.040, 0.040, SKIN)
        add(shoulder_r, right_dir, upper_len, self._n(5), 0.050, 0.050, SHIRT)
        add(elbow_r, fore_r, fore_len, self._n(5), 0.040, 0.040, SKIN)
        add(hip_l, thigh_l, thigh_len, self._n(6), 0.075, 0.075, PANTS)
        add(knee_l, shin_l, shin_len, self._n(6), 0.055, 0.055, PANTS)
        add(hip_r, thigh_r, thigh_len, self._n(6), 0.075, 0.075, PANTS)
        add(knee_r, shin_r, shin_len, self._n(6), 0.055, 0.055, PANTS)
        foot_l = knee_l + shin_l * shin_len
        foot_r = knee_r + shin_r * shin_len
        add(foot_l + pose.new_tensor([0.0, 0.02, 0.04]), pose.new_tensor([0.0, 0.0, 1.0]), 0.16, self._n(3), 0.045, 0.03, SHOE)
        add(foot_r + pose.new_tensor([0.0, 0.02, 0.04]), pose.new_tensor([0.0, 0.0, 1.0]), 0.16, self._n(3), 0.045, 0.03, SHOE)

        head_center = pose.new_tensor([0.0, 1.60, 0.02])
        head_n = self._n(70)
        index = torch.arange(head_n, device=device, dtype=dtype)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        y = 1.0 - (index / max(head_n - 1, 1)) * 2.0
        ring = torch.sqrt((1.0 - y.square()).clamp(min=0))
        head = torch.stack(
            [ring * torch.cos(golden * index), y, ring * torch.sin(golden * index)],
            dim=1,
        )
        head_radius = 0.115
        head_means = head_center + head * head_radius
        add_cloud(head_means, head_radius * 0.42, SKIN)
        hair = head_means[y > 0.15] + pose.new_tensor([0.0, 0.02, -0.02])
        add_cloud(hair, head_radius * 0.36, HAIR)
        eyes = torch.stack(
            [
                head_center + pose.new_tensor([-0.038, 0.015, 0.10]),
                head_center + pose.new_tensor([0.038, 0.015, 0.10]),
            ]
        )
        add_cloud(eyes, 0.018, EYE)
        return (
            torch.cat(means_all, 0),
            torch.cat(rot_all, 0),
            torch.cat(scale_all, 0),
            torch.cat(color_all, 0),
        )


def dense_body_gaussians(n: int, device: torch.device, seed: int = 0):
    """Repeat the body with jitter until there are n Gaussians."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    avatar = HumanAvatar(density=1.0)
    means, cov, color, _opacity = avatar.gaussians(rest_pose(), template=True)
    chunks_m, chunks_c, chunks_k = [], [], []
    remaining = n
    while remaining > 0:
        take = min(remaining, means.shape[0])
        jitter = torch.randn(take, 3, generator=generator) * 0.012
        chunks_m.append(means[:take] + jitter)
        chunks_c.append(color[:take])
        chunks_k.append(cov[:take])
        remaining -= take
    opacity = torch.full((n,), 0.85, device=device)
    return (
        torch.cat(chunks_m, 0).to(device),
        torch.cat(chunks_k, 0).to(device),
        torch.cat(chunks_c, 0).to(device),
        opacity,
    )
