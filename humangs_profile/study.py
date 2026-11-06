from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from humangs_profile.body import POSE_DIM, HumanAvatar, dense_body_gaussians, rest_pose, sample_pose
from humangs_profile.diffusion import (
    Denoiser,
    condition_vector,
    make_schedule,
    q_sample,
    sds_surrogate,
)
from humangs_profile.render import render, self_check


TEST_YAW = 0.0
TEST_ELEV = 0.0
HOLD_OUT = 0.45


def angular_distance(yaw: torch.Tensor, center: float) -> torch.Tensor:
    return torch.atan2(torch.sin(yaw - center), torch.cos(yaw - center)).abs()


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = (prediction.clamp(0, 1) - target.clamp(0, 1)).square().mean().clamp(min=1e-10)
    return float((10.0 * torch.log10(1.0 / mse)).item())


def to_model(image: torch.Tensor) -> torch.Tensor:
    """HWC in [0, 1] to NCHW in [-1, 1]."""
    if image.ndim == 3:
        image = image[None]
    return image.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0


def from_model(image: torch.Tensor) -> torch.Tensor:
    return (image.clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1)


def cuda_ms(fn, repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def save_png(path: Path, image: torch.Tensor):
    from PIL import Image

    array = (image.detach().clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(array).save(path)


def make_panel(path: Path, frames: list[tuple[str, torch.Tensor]]):
    from PIL import Image, ImageDraw

    width = frames[0][1].shape[1]
    height = frames[0][1].shape[0]
    label = 18
    canvas = Image.new("RGB", (width * len(frames), height + label), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(frames):
        array = (image.detach().clamp(0, 1) * 255).byte().cpu().numpy()
        tile = Image.fromarray(array).resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(tile, (index * width, label))
        draw.text((index * width + 4, 2), name, fill=(0, 0, 0))
    canvas.save(path)


def sample_training_batch(avatar, batch, height, generator):
    device = avatar.template_color.device
    yaw = torch.rand(batch, device=device, generator=generator) * (2 * math.pi)
    # Reject the held-out front arc so the test view is not a memorized frame.
    for _ in range(8):
        bad = angular_distance(yaw, TEST_YAW) < HOLD_OUT
        if not bool(bad.any()):
            break
        yaw = torch.where(
            bad,
            torch.rand(batch, device=device, generator=generator) * (2 * math.pi),
            yaw,
        )
    elev = (torch.rand(batch, device=device, generator=generator) * 2 - 1) * 0.18
    pose = sample_pose(batch, device=device, generator=generator)
    images = []
    for index in range(batch):
        means, cov, color, opacity = avatar.gaussians(pose[index], template=True)
        images.append(
            render(
                means, cov, color, opacity,
                float(yaw[index]), float(elev[index]), height, height,
            )
        )
    stack = torch.stack(images, 0)
    return to_model(stack), condition_vector(yaw, elev, pose)


@torch.no_grad()
def ddim_sample(model, cond, schedule, steps, shape):
    device = cond.device
    total = int(schedule["alpha_bar"].shape[0])
    choose = torch.linspace(total - 1, 0, steps, device=device).long()
    x = torch.randn(shape, device=device)
    alpha_bar = schedule["alpha_bar"]
    for cursor, t_value in enumerate(choose):
        t = torch.full((shape[0],), int(t_value), device=device, dtype=torch.long)
        epsilon = model(x, t, cond)
        bar = alpha_bar[t][:, None, None, None]
        x0 = ((x - (1 - bar).sqrt() * epsilon) / bar.sqrt()).clamp(-1, 1)
        if cursor == len(choose) - 1:
            x = x0
            break
        previous = int(choose[cursor + 1])
        bar_prev = alpha_bar[previous]
        x = bar_prev.sqrt() * x0 + (1 - bar_prev).sqrt() * epsilon
    return x


def fit_avatar(avatar, target, yaw, elev, height, steps, optimizer, mode, model, schedule, pose):
    """mode is 'l2' or 'sds'. Returns PSNR after the last step."""
    device = target.device
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        means, cov, color, opacity = avatar.gaussians(pose, template=False)
        image = render(means, cov, color, opacity, yaw, elev, height, height)
        image_psnr = psnr(image, target)
        if mode == "l2":
            loss = (image - target).square().mean()
        else:
            x0 = to_model(image)
            t = torch.randint(20, schedule["alpha_bar"].shape[0] - 20, (1,), device=device)
            epsilon = torch.randn_like(x0)
            noisy = q_sample(x0, t, epsilon, schedule["alpha_bar"])
            cond = condition_vector(
                torch.tensor([yaw], device=device),
                torch.tensor([elev], device=device),
                pose[None],
            )
            prediction = model(noisy, t, cond).detach()
            loss = sds_surrogate(x0, t, prediction, epsilon, schedule["alpha_bar"])
            # Keep the silhouette on the skeleton. Appearance is not anchored.
            loss = loss + 0.02 * avatar.mean_residual.square().mean()
            loss = loss + 0.001 * avatar.log_scale.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(avatar.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step, "psnr": image_psnr, "loss": float(loss.detach())})
    return history


def benchmark(device, repeats: int) -> list[dict]:
    rows = []
    for n, size in ((avatar_count_hint(device), 64), (1024, 128), (4096, 192)):
        means, cov, color, opacity = dense_body_gaussians(n, device)
        means = means.requires_grad_(True)
        color = color.detach().requires_grad_(True)

        def forward(backend, _means=means, _cov=cov, _color=color, _opacity=opacity, _size=size):
            with torch.no_grad():
                render(_means, _cov, _color, _opacity, TEST_YAW, TEST_ELEV, _size, _size, backend)

        def forward_backward(backend, _means=means, _cov=cov, _color=color, _opacity=opacity, _size=size):
            image = render(_means, _cov, _color, _opacity, TEST_YAW, TEST_ELEV, _size, _size, backend)
            image.square().mean().backward()

        row = {"gaussians": n, "resolution": size}
        row["cuda_forward_ms"] = cuda_ms(lambda: forward("cuda"), repeats)
        row["cuda_forward_backward_ms"] = cuda_ms(lambda: forward_backward("cuda"), repeats)
        # Reference is one kernel per Gaussian. Time it only at size <= 64.
        if size <= 64:
            row["reference_forward_ms"] = cuda_ms(lambda: forward("reference"), max(2, repeats // 2), warmup=1)
            row["reference_forward_backward_ms"] = cuda_ms(
                lambda: forward_backward("reference"), max(2, repeats // 2), warmup=1
            )
        rows.append(row)
        means.requires_grad_(False)
        color.requires_grad_(False)
    return rows


def avatar_count_hint(device) -> int:
    avatar = HumanAvatar(density=2.0).to(device)
    return avatar.count


def copy_avatar(source: HumanAvatar) -> HumanAvatar:
    clone = HumanAvatar(density=source.density).to(source.template_color.device)
    clone.load_state_dict(source.state_dict())
    return clone


def appearance_optimizer(avatar: HumanAvatar) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": [avatar.color_logit], "lr": 0.08},
            {"params": [avatar.opacity_logit], "lr": 0.05},
            {"params": [avatar.log_scale], "lr": 0.01},
            {"params": [avatar.mean_residual], "lr": 0.002},
        ]
    )


def train_prior(avatar, model, schedule, steps, batch, height, generator, log_every):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        x0, cond = sample_training_batch(avatar, batch, height, generator)
        t = torch.randint(0, schedule["alpha_bar"].shape[0], (batch,), device=x0.device)
        epsilon = torch.randn_like(x0)
        noisy = q_sample(x0, t, epsilon, schedule["alpha_bar"])
        prediction = model(noisy, t, cond)
        loss = (prediction - epsilon).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            print(f"  prior step {step:4d}  mse {losses[-1]:.4f}", flush=True)
    return losses, time.perf_counter() - t0


def profile_sds_step(avatar, model, schedule, pose, yaw, elev, height) -> dict[str, float]:
    """Time each piece of one score-distillation step with CUDA events."""
    device = avatar.template_color.device
    model.eval()

    def render_only():
        with torch.no_grad():
            means, cov, color, opacity = avatar.gaussians(pose, template=False)
            render(means, cov, color, opacity, yaw, elev, height, height)

    def denoiser_only():
        with torch.no_grad():
            x = torch.randn(1, 3, height, height, device=device)
            t = torch.full((1,), schedule["alpha_bar"].shape[0] // 2, device=device, dtype=torch.long)
            cond = condition_vector(
                torch.tensor([yaw], device=device),
                torch.tensor([elev], device=device),
                pose[None],
            )
            model(x, t, cond)

    def render_backward():
        means, cov, color, opacity = avatar.gaussians(pose, template=False)
        image = render(means, cov, color, opacity, yaw, elev, height, height)
        image.square().mean().backward()
        avatar.zero_grad(set_to_none=True)

    return {
        "render_forward_ms": cuda_ms(render_only, repeats=10),
        "denoiser_forward_ms": cuda_ms(denoiser_only, repeats=10),
        "render_forward_backward_ms": cuda_ms(render_backward, repeats=8, warmup=2),
    }


def write_timing_figure(path: Path, rows: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"N={row['gaussians']}\n{row['resolution']}^2" for row in rows]
    cuda_fwd = [row["cuda_forward_ms"] for row in rows]
    ref_fwd = [row.get("reference_forward_ms", 0.0) for row in rows]
    x = range(len(rows))
    plt.figure(figsize=(7.2, 3.6))
    plt.bar([i - 0.18 for i in x], cuda_fwd, width=0.36, label="tiled CUDA")
    plt.bar([i + 0.18 for i in x], ref_fwd, width=0.36, label="PyTorch loop")
    plt.xticks(list(x), labels)
    plt.ylabel("forward milliseconds")
    plt.title("Rasterizer forward time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def run(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("checking the tiled rasterizer against the PyTorch reference", flush=True)
    match = self_check(device)
    print(
        f"  forward max abs {match['forward_max_abs']:.3e}  "
        f"grad max abs {max(match['grad_max_abs']):.3e}",
        flush=True,
    )

    avatar = HumanAvatar(density=2.0).to(device)
    pose = rest_pose(device)
    height = args.image_size
    print(f"subject has {avatar.count} Gaussians", flush=True)

    with torch.no_grad():
        means, cov, color, opacity = avatar.gaussians(pose, template=True)
        ground_truth = render(means, cov, color, opacity, TEST_YAW, TEST_ELEV, height, height)
        hero = render(means, cov, color, opacity, TEST_YAW, TEST_ELEV, 256, 256)
    save_png(out / "ground_truth_64.png", ground_truth)
    save_png(out / "ground_truth_256.png", hero)

    steps = 30 if args.quick else args.diffusion_steps
    sds_steps = 4 if args.quick else args.sds_steps
    batch = 4 if args.quick else args.batch_size
    diffusion_steps_t = 50 if args.quick else 200
    schedule = make_schedule(diffusion_steps_t, device)
    model = Denoiser(cond_dim=4 + POSE_DIM).to(device)
    print(f"training the pose-conditioned prior for {steps} steps", flush=True)
    losses, prior_seconds = train_prior(
        avatar, model, schedule, steps, batch, height, generator, log_every=max(1, steps // 8)
    )
    model.eval()

    with torch.no_grad():
        cond = condition_vector(
            torch.tensor([TEST_YAW, 1.2, 2.4, 3.6], device=device),
            torch.zeros(4, device=device),
            pose.expand(4, -1),
        )
        samples = from_model(ddim_sample(model, cond, schedule, steps=20 if args.quick else 40, shape=(4, 3, height, height)))
    for index in range(samples.shape[0]):
        save_png(out / f"prior_sample_{index}.png", samples[index])

    target = ground_truth.detach()
    init_avatar = copy_avatar(avatar)
    with torch.no_grad():
        means, cov, color, opacity = init_avatar.gaussians(pose, template=False)
        initial = render(means, cov, color, opacity, TEST_YAW, TEST_ELEV, height, height)
    save_png(out / "init.png", initial)

    print(f"fitting appearance with score distillation for {sds_steps} steps", flush=True)
    sds_avatar = copy_avatar(avatar)
    sds_history = fit_avatar(
        sds_avatar, target, TEST_YAW, TEST_ELEV, height, sds_steps,
        appearance_optimizer(sds_avatar), "sds", model, schedule, pose,
    )
    with torch.no_grad():
        means, cov, color, opacity = sds_avatar.gaussians(pose, template=False)
        sds_image = render(means, cov, color, opacity, TEST_YAW, TEST_ELEV, height, height)
        sds_hero = render(means, cov, color, opacity, TEST_YAW, TEST_ELEV, 256, 256)
    save_png(out / "sds.png", sds_image)
    save_png(out / "sds_256.png", sds_hero)

    print(f"fitting appearance with held-out-view L2 for {sds_steps} steps", flush=True)
    l2_avatar = copy_avatar(avatar)
    l2_history = fit_avatar(
        l2_avatar, target, TEST_YAW, TEST_ELEV, height, sds_steps,
        appearance_optimizer(l2_avatar), "l2", model, schedule, pose,
    )
    with torch.no_grad():
        means, cov, color, opacity = l2_avatar.gaussians(pose, template=False)
        l2_image = render(means, cov, color, opacity, TEST_YAW, TEST_ELEV, height, height)
    save_png(out / "l2.png", l2_image)

    make_panel(
        out / "panel.png",
        [
            ("ground truth", ground_truth),
            ("gray init", initial),
            ("diffusion SDS", sds_image),
            ("L2 on view", l2_image),
            ("prior sample", samples[0]),
        ],
    )

    print("timing the rasterizer and one distillation step", flush=True)
    rows = benchmark(device, repeats=3 if args.quick else 8)
    write_timing_figure(out / "raster_time.png", rows)
    breakdown = profile_sds_step(sds_avatar, model, schedule, pose, TEST_YAW, TEST_ELEV, height)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "gaussians": avatar.count,
        "image_size": height,
        "diffusion_steps": steps,
        "diffusion_loss_start": losses[0],
        "diffusion_loss_end": losses[-1],
        "prior_seconds": prior_seconds,
        "sds_steps": sds_steps,
        "psnr_init": psnr(initial, target),
        "psnr_sds": psnr(sds_image, target),
        "psnr_l2": psnr(l2_image, target),
        "raster_match": match,
        "raster_benchmark": rows,
        "sds_breakdown_ms": breakdown,
        "sds_history_tail": sds_history[-3:],
        "l2_history_tail": l2_history[-3:],
        "held_out_yaw_radians": HOLD_OUT,
    }
    (out / "timings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=800)
    parser.add_argument("--sds-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this study needs a CUDA GPU")
    run(parse_args())


if __name__ == "__main__":
    main()
