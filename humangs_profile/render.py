from __future__ import annotations

import math
import os
from pathlib import Path

import torch

TILE = 16
MAX_CONTRIB = 128
MAX_TILE_SPAN = 12
ALPHA_THRESH = 1.0 / 255.0


def look_at(yaw: float, elev: float, distance: float, device, dtype):
    """Camera on a ring, looking at the torso. Rows of R are right, up, forward."""
    eye = torch.tensor(
        [
            distance * math.sin(yaw) * math.cos(elev),
            0.85 + distance * math.sin(elev),
            distance * math.cos(yaw) * math.cos(elev),
        ],
        device=device,
        dtype=dtype,
    )
    target = torch.tensor([0.0, 0.85, 0.0], device=device, dtype=dtype)
    forward = target - eye
    forward = forward / forward.norm()
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    cam_up = torch.linalg.cross(right, forward)
    rotation = torch.stack([right, cam_up, forward], dim=0)
    return eye, rotation


def project(means, cov, yaw, elev, height, width, distance=3.15, fov_deg=32.0):
    """Project world Gaussians to pixel-space means and a 2D conic.

    Returns xy (N, 2), conic (N, 3), depth (N,), and a validity mask.
    `conic` is (a, b, c) for [[a, b], [b, c]], the inverse of the screen covariance.
    """
    device, dtype = means.device, means.dtype
    eye, rotation = look_at(yaw, elev, distance, device, dtype)
    focal = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
    cx = width * 0.5
    cy = height * 0.5

    relative = means - eye
    camera = relative @ rotation.transpose(0, 1)
    depth = camera[:, 2]
    z = depth.clamp(min=1e-2)
    x = camera[:, 0]
    y = camera[:, 1]
    xy = torch.stack([focal * x / z + cx, cy - focal * y / z], dim=1)

    jacobian = means.new_zeros(means.shape[0], 2, 3)
    jacobian[:, 0, 0] = focal / z
    jacobian[:, 0, 2] = -focal * x / z.square()
    jacobian[:, 1, 1] = -focal / z
    jacobian[:, 1, 2] = focal * y / z.square()

    rotation_b = rotation.expand(means.shape[0], 3, 3)
    cov_cam = rotation_b @ cov @ rotation_b.transpose(1, 2)
    cov_2d = jacobian @ cov_cam @ jacobian.transpose(1, 2)
    cov_2d = cov_2d.clone()
    # The same low-pass 0.3 pixel term used by 3D Gaussian Splatting.
    cov_2d[:, 0, 0] = cov_2d[:, 0, 0] + 0.3
    cov_2d[:, 1, 1] = cov_2d[:, 1, 1] + 0.3
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    c = cov_2d[:, 1, 1]
    det = a * c - b.square()
    valid = (depth > 0.15) & (det > 1e-6)
    inv_det = torch.where(valid, 1.0 / det.clamp(min=1e-6), torch.zeros_like(det))
    conic = torch.stack([c * inv_det, -b * inv_det, a * inv_det], dim=1)
    return xy, conic, depth, valid


def support_radius(conic: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    """Pixel radius that covers every sample with alpha >= 1/255."""
    a, b, c = conic[:, 0], conic[:, 1], conic[:, 2]
    det = a * c - b.square()
    valid = det > 1e-8
    inv_det = torch.where(valid, 1.0 / det.clamp(min=1e-8), torch.zeros_like(det))
    cov_a = c * inv_det
    cov_b = -b * inv_det
    cov_c = a * inv_det
    disc = (cov_a - cov_c).square() + 4.0 * cov_b.square()
    lam = (0.5 * (cov_a + cov_c + disc.clamp(min=0).sqrt())).clamp(min=0)
    visible = valid & (opacity > ALPHA_THRESH)
    ratio = (ALPHA_THRESH / opacity.clamp(min=ALPHA_THRESH + 1e-8)).clamp(max=1.0)
    quad = -2.0 * torch.log(ratio.clamp(min=1e-8))
    radius = torch.sqrt((lam * quad).clamp(min=0)) + 1.0
    return torch.where(visible, radius, torch.zeros_like(radius))


def build_tiles(xy, radius, depth, height, width):
    """Bin Gaussians into tiles. Bins are detached."""
    device = xy.device
    n = xy.shape[0]
    tiles_x = (width + TILE - 1) // TILE
    tiles_y = (height + TILE - 1) // TILE
    n_tiles = tiles_x * tiles_y
    if n == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        ranges = torch.zeros(n_tiles, 2, dtype=torch.int32, device=device)
        return empty, ranges

    xy_d = xy.detach()
    radius_d = radius.detach()
    min_tx = torch.div(xy_d[:, 0] - radius_d, TILE, rounding_mode="floor").int().clamp(0, tiles_x - 1)
    max_tx = torch.div(xy_d[:, 0] + radius_d, TILE, rounding_mode="floor").int().clamp(0, tiles_x - 1)
    min_ty = torch.div(xy_d[:, 1] - radius_d, TILE, rounding_mode="floor").int().clamp(0, tiles_y - 1)
    max_ty = torch.div(xy_d[:, 1] + radius_d, TILE, rounding_mode="floor").int().clamp(0, tiles_y - 1)
    span_x = (max_tx - min_tx + 1).clamp(min=0)
    span_y = (max_ty - min_ty + 1).clamp(min=0)
    alive = radius_d > 0
    span_x = torch.where(alive, span_x, torch.zeros_like(span_x))
    span_y = torch.where(alive, span_y, torch.zeros_like(span_y))
    span = int(max(span_x.max().item(), span_y.max().item(), 1))
    if span > MAX_TILE_SPAN:
        raise RuntimeError(
            f"a Gaussian covers {span} tiles (cap {MAX_TILE_SPAN}). Reduce its scale."
        )

    offset = torch.arange(span, device=device)
    tx = min_tx[:, None, None] + offset[None, None, :]
    ty = min_ty[:, None, None] + offset[None, :, None]
    valid = (offset[None, None, :] < span_x[:, None, None]) & (
        offset[None, :, None] < span_y[:, None, None]
    )
    valid = valid & (tx < tiles_x) & (ty < tiles_y)
    tile_id = (ty * tiles_x + tx)[valid].to(torch.int64)
    gauss_id = (
        torch.arange(n, device=device)[:, None, None].expand(n, span, span)[valid].to(torch.int64)
    )
    if tile_id.numel() == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        ranges = torch.zeros(n_tiles, 2, dtype=torch.int32, device=device)
        return empty, ranges

    order = torch.argsort(depth.detach())
    rank = torch.empty(n, dtype=torch.int64, device=device)
    rank[order] = torch.arange(n, device=device)
    key = tile_id * (n + 1) + rank[gauss_id]
    perm = torch.argsort(key)
    tile_id = tile_id[perm]
    gauss_id = gauss_id[perm].to(torch.int32)
    counts = torch.bincount(tile_id, minlength=n_tiles)
    ends = torch.cumsum(counts, 0)
    starts = ends - counts
    ranges = torch.stack([starts, ends], dim=1).to(torch.int32)
    return gauss_id, ranges


def composite_reference(xy, conic, opacity, color, depth, height, width):
    order = torch.argsort(depth.detach())
    acc = xy.new_zeros(height, width, 3)
    transmittance = xy.new_ones(height, width)
    ys = torch.arange(height, device=xy.device, dtype=xy.dtype) + 0.5
    xs = torch.arange(width, device=xy.device, dtype=xy.dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    for index in order.tolist():
        dx = grid_x - xy[index, 0]
        dy = grid_y - xy[index, 1]
        a, b, c = conic[index, 0], conic[index, 1], conic[index, 2]
        power = -0.5 * (a * dx.square() + c * dy.square()) - b * dx * dy
        raw = opacity[index] * torch.exp(power)
        raw = torch.where(power > 0, torch.zeros_like(raw), raw)
        alpha = torch.where(raw < ALPHA_THRESH, torch.zeros_like(raw), raw).clamp(max=0.99)
        alpha = torch.where(transmittance > 1e-4, alpha, torch.zeros_like(alpha))
        weight = transmittance * alpha
        acc = acc + weight[..., None] * color[index]
        transmittance = transmittance * (1.0 - alpha)
    return acc + transmittance[..., None]


_CUDA = None


def load_cuda():
    global _CUDA
    if _CUDA is not None:
        return _CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("the tiled rasterizer needs a CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    # Torch looks up distutils._msvccompiler as an attribute. Setuptools 75+
    # only creates that attribute once the submodule has been imported.
    import distutils._msvccompiler  # noqa: F401
    from torch.utils.cpp_extension import load

    source = Path(__file__).parent / "csrc"
    (source / "build").mkdir(parents=True, exist_ok=True)
    _CUDA = load(
        name="humangs_composite",
        sources=[str(source / "composite.cpp"), str(source / "composite.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        build_directory=str(source / "build"),
        with_cuda=True,
        verbose=False,
    )
    return _CUDA


class Composite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, conic, opacity, color, depth, height, width):
        radius = support_radius(conic, opacity)
        gauss_ids, ranges = build_tiles(xy, radius, depth, int(height), int(width))
        module = load_cuda()
        image, ids, alpha, trans, clamped, count, overflow = module.forward(
            xy.contiguous(),
            conic.contiguous(),
            opacity.contiguous(),
            color.contiguous(),
            gauss_ids.contiguous(),
            ranges.contiguous(),
            int(height),
            int(width),
            TILE,
            MAX_CONTRIB,
        )
        if int(overflow.item()) != 0:
            raise RuntimeError(
                f"a pixel overlapped more than {MAX_CONTRIB} Gaussians "
                f"({int(overflow.item())} dropped samples). Raise MAX_CONTRIB."
            )
        ctx.save_for_backward(xy, conic, opacity, color, ids, alpha, trans, clamped, count)
        ctx.max_contrib = MAX_CONTRIB
        return image

    @staticmethod
    def backward(ctx, grad_out):
        xy, conic, opacity, color, ids, alpha, trans, clamped, count = ctx.saved_tensors
        module = load_cuda()
        g_xy, g_conic, g_opacity, g_color = module.backward(
            grad_out.contiguous(),
            xy.contiguous(),
            conic.contiguous(),
            opacity.contiguous(),
            color.contiguous(),
            ids,
            alpha,
            trans,
            clamped,
            count,
            ctx.max_contrib,
        )
        return g_xy, g_conic, g_opacity, g_color, None, None, None


def render(means, cov, color, opacity, yaw, elev, height, width, backend="cuda"):
    xy, conic, depth, valid = project(means, cov, yaw, elev, height, width)
    opacity = torch.where(valid, opacity, torch.zeros_like(opacity))
    if backend == "cuda":
        image = Composite.apply(xy, conic, opacity, color, depth, height, width)
    elif backend == "reference":
        image = composite_reference(xy, conic, opacity, color, depth, height, width)
    else:
        raise ValueError(backend)
    return image


def self_check(device=None):
    """Fail loud if the CUDA kernel drifts from the PyTorch composite."""
    device = device or torch.device("cuda")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    n, height, width = 24, 40, 40
    xy = torch.rand(n, 2, generator=generator) * torch.tensor([width, height])
    variance = 2.5 + torch.rand(n, generator=generator) * 4.0
    angle = torch.rand(n, generator=generator) * math.pi
    # Build a PSD conic from an anisotropic covariance so the radius is finite.
    major = variance
    minor = variance * (0.35 + 0.4 * torch.rand(n, generator=generator))
    cos, sin = torch.cos(angle), torch.sin(angle)
    cov_a = major * cos.square() + minor * sin.square()
    cov_b = (major - minor) * cos * sin
    cov_c = major * sin.square() + minor * cos.square()
    det = (cov_a * cov_c - cov_b.square()).clamp(min=1e-4)
    conic = torch.stack([cov_c / det, -cov_b / det, cov_a / det], dim=1)
    opacity = 0.25 + 0.6 * torch.rand(n, generator=generator)
    color = torch.rand(n, 3, generator=generator)
    depth = torch.arange(n, dtype=torch.float32)
    tensors = [xy, conic, opacity, color, depth]
    tensors = [t.to(device) for t in tensors]
    xy, conic, opacity, color, depth = tensors
    for leaf in (xy, conic, opacity, color):
        leaf.requires_grad_(True)
    ref = composite_reference(xy, conic, opacity, color, depth, height, width)
    cuda = Composite.apply(xy, conic, opacity, color, depth, height, width)
    forward_error = (ref - cuda).abs().max().item()
    loss_ref = ref.square().mean()
    loss_cuda = cuda.square().mean()
    grads_ref = torch.autograd.grad(loss_ref, (xy, conic, opacity, color), retain_graph=True)
    grads_cuda = torch.autograd.grad(loss_cuda, (xy, conic, opacity, color))
    grad_errors = [(a - b).abs().max().item() for a, b in zip(grads_ref, grads_cuda)]
    if forward_error > 2e-3 or max(grad_errors) > 5e-3:
        raise AssertionError(
            f"rasterizer mismatch forward {forward_error:.3e} grads {grad_errors}"
        )
    return {"forward_max_abs": forward_error, "grad_max_abs": grad_errors}
