// One block per 16x16 tile. Shared chunk of primitives, front-to-back.
// alpha = opacity * exp(-0.5 * quad), quad = [dx, dy] Conic [dx, dy].
// Conic is (a, b, c) for [[a, b], [b, c]].
// dC/dalpha_i = T_i * (color_i - color_behind_i).

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kChunk = 64;
constexpr float kAlphaThresh = 1.0f / 255.0f;
constexpr float kAlphaMax = 0.99f;

struct Prim {
    float2 xy;
    float3 conic;
    float opacity;
    float3 color;
};

__device__ __forceinline__ Prim load_prim(
    const float* __restrict__ xy,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    int id) {
    Prim p;
    p.xy = make_float2(xy[id * 2], xy[id * 2 + 1]);
    p.conic = make_float3(conic[id * 3], conic[id * 3 + 1], conic[id * 3 + 2]);
    p.opacity = opacity[id];
    p.color = make_float3(color[id * 3], color[id * 3 + 1], color[id * 3 + 2]);
    return p;
}

template <int TILE>
__global__ void composite_forward_kernel(
    const float* __restrict__ xy,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const int* __restrict__ gauss_ids,
    const int* __restrict__ tile_ranges,
    float* __restrict__ out,
    int* __restrict__ contrib_ids,
    float* __restrict__ contrib_alpha,
    float* __restrict__ contrib_T,
    std::uint8_t* __restrict__ contrib_clamped,
    int* __restrict__ contrib_count,
    int* __restrict__ overflow,
    int height,
    int width,
    int max_contrib) {
    const int tile_x = blockIdx.x;
    const int tile_y = blockIdx.y;
    const int tiles_x = gridDim.x;
    const int pix_x = tile_x * TILE + threadIdx.x;
    const int pix_y = tile_y * TILE + threadIdx.y;
    const int tile_id = tile_y * tiles_x + tile_x;
    const int start = tile_ranges[tile_id * 2];
    const int end = tile_ranges[tile_id * 2 + 1];
    const bool inside = pix_x < width && pix_y < height;
    const int tid = threadIdx.y * TILE + threadIdx.x;

    __shared__ Prim smem[kChunk];

    float transmittance = 1.f;
    float red = 0.f, green = 0.f, blue = 0.f;
    int stored = 0;
    const float pcx = static_cast<float>(pix_x) + 0.5f;
    const float pcy = static_cast<float>(pix_y) + 0.5f;

    for (int base = start; base < end; base += kChunk) {
        const int remain = end - base;
        const int nload = remain < kChunk ? remain : kChunk;
        for (int j = tid; j < nload; j += TILE * TILE) {
            smem[j] = load_prim(xy, conic, opacity, color, gauss_ids[base + j]);
        }
        __syncthreads();

        if (inside) {
            for (int j = 0; j < nload; ++j) {
                // Stop when the pixel is opaque. Matches the reference.
                if (transmittance < 1e-4f) {
                    break;
                }
                const Prim p = smem[j];
                const float dx = pcx - p.xy.x;
                const float dy = pcy - p.xy.y;
                const float quad =
                    p.conic.x * dx * dx + p.conic.z * dy * dy + 2.f * p.conic.y * dx * dy;
                const float power = -0.5f * quad;
                if (power > 0.f) {
                    continue;
                }
                const float raw = p.opacity * expf(power);
                if (raw < kAlphaThresh) {
                    continue;
                }
                const bool clamped = raw > kAlphaMax;
                const float alpha = clamped ? kAlphaMax : raw;
                if (stored < max_contrib) {
                    const int pix = pix_y * width + pix_x;
                    const int slot = pix * max_contrib + stored;
                    contrib_ids[slot] = gauss_ids[base + j];
                    contrib_alpha[slot] = alpha;
                    contrib_T[slot] = transmittance;
                    contrib_clamped[slot] = static_cast<std::uint8_t>(clamped);
                    stored += 1;
                } else {
                    atomicAdd(overflow, 1);
                }
                const float weight = transmittance * alpha;
                red += weight * p.color.x;
                green += weight * p.color.y;
                blue += weight * p.color.z;
                transmittance *= (1.f - alpha);
            }
        }
        __syncthreads();
    }

    if (inside) {
        const int pix = pix_y * width + pix_x;
        // Background is white. color_behind of the farthest splat starts here.
        out[pix * 3] = red + transmittance;
        out[pix * 3 + 1] = green + transmittance;
        out[pix * 3 + 2] = blue + transmittance;
        contrib_count[pix] = stored;
    }
}

__global__ void composite_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ xy,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const int* __restrict__ contrib_ids,
    const float* __restrict__ contrib_alpha,
    const float* __restrict__ contrib_T,
    const std::uint8_t* __restrict__ contrib_clamped,
    const int* __restrict__ contrib_count,
    const float* __restrict__ color,
    float* __restrict__ g_xy,
    float* __restrict__ g_conic,
    float* __restrict__ g_opacity,
    float* __restrict__ g_color,
    int height,
    int width,
    int max_contrib) {
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= height * width) {
        return;
    }
    const int count = contrib_count[pix];
    if (count <= 0) {
        return;
    }
    const int py = pix / width;
    const int px = pix - py * width;
    const float pcx = static_cast<float>(px) + 0.5f;
    const float pcy = static_cast<float>(py) + 0.5f;
    const float dcr = grad_out[pix * 3];
    const float dcg = grad_out[pix * 3 + 1];
    const float dcb = grad_out[pix * 3 + 2];

    // Color behind the splat currently being visited, including the white background.
    float behind_r = 1.f, behind_g = 1.f, behind_b = 1.f;

    for (int k = count - 1; k >= 0; --k) {
        const int slot = pix * max_contrib + k;
        const int id = contrib_ids[slot];
        const float alpha = contrib_alpha[slot];
        const float transmittance = contrib_T[slot];
        const bool clamped = contrib_clamped[slot] != 0;

        const float cr = color[id * 3];
        const float cg = color[id * 3 + 1];
        const float cb = color[id * 3 + 2];
        const float d_alpha = transmittance * (
            dcr * (cr - behind_r) + dcg * (cg - behind_g) + dcb * (cb - behind_b));
        const float weight = transmittance * alpha;
        atomicAdd(g_color + id * 3, dcr * weight);
        atomicAdd(g_color + id * 3 + 1, dcg * weight);
        atomicAdd(g_color + id * 3 + 2, dcb * weight);

        behind_r = alpha * cr + (1.f - alpha) * behind_r;
        behind_g = alpha * cg + (1.f - alpha) * behind_g;
        behind_b = alpha * cb + (1.f - alpha) * behind_b;

        if (clamped || d_alpha == 0.f) {
            continue;
        }

        const float a = conic[id * 3];
        const float b = conic[id * 3 + 1];
        const float c = conic[id * 3 + 2];
        const float dx = pcx - xy[id * 2];
        const float dy = pcy - xy[id * 2 + 1];
        const float op = opacity[id];
        // Stored alpha is opacity * exp(power) when the sample was not clamped.
        const float d_power = d_alpha * alpha;
        const float exp_power = (op > 1e-8f) ? (alpha / op) : 0.f;
        atomicAdd(g_opacity + id, d_alpha * exp_power);
        atomicAdd(g_conic + id * 3, d_power * (-0.5f * dx * dx));
        atomicAdd(g_conic + id * 3 + 1, d_power * (-dx * dy));
        atomicAdd(g_conic + id * 3 + 2, d_power * (-0.5f * dy * dy));
        const float d_dx = d_power * (-(a * dx + b * dy));
        const float d_dy = d_power * (-(b * dx + c * dy));
        atomicAdd(g_xy + id * 2, -d_dx);
        atomicAdd(g_xy + id * 2 + 1, -d_dy);
    }
}

}  // namespace

std::vector<torch::Tensor> composite_forward_cuda(
    torch::Tensor xy,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor gauss_ids,
    torch::Tensor tile_ranges,
    int64_t height,
    int64_t width,
    int64_t tile,
    int64_t max_contrib) {
    TORCH_CHECK(xy.is_cuda(), "xy must be a CUDA tensor");
    TORCH_CHECK(tile == 16, "this kernel is compiled for 16x16 tiles");
    TORCH_CHECK(max_contrib > 0 && max_contrib <= 160, "max_contrib out of range");

    auto opts_f = xy.options();
    auto opts_i = xy.options().dtype(torch::kInt32);
    auto opts_u8 = xy.options().dtype(torch::kUInt8);
    auto image = torch::ones({height, width, 3}, opts_f);
    auto contrib_ids = torch::zeros({height, width, max_contrib}, opts_i);
    auto contrib_alpha = torch::zeros({height, width, max_contrib}, opts_f);
    auto contrib_T = torch::zeros({height, width, max_contrib}, opts_f);
    auto contrib_clamped = torch::zeros({height, width, max_contrib}, opts_u8);
    auto contrib_count = torch::zeros({height, width}, opts_i);
    auto overflow = torch::zeros({}, opts_i);

    const int tiles_x = static_cast<int>((width + tile - 1) / tile);
    const int tiles_y = static_cast<int>((height + tile - 1) / tile);
    dim3 block(static_cast<unsigned>(tile), static_cast<unsigned>(tile));
    dim3 grid(tiles_x, tiles_y);
    composite_forward_kernel<16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        xy.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        gauss_ids.data_ptr<int>(),
        tile_ranges.data_ptr<int>(),
        image.data_ptr<float>(),
        contrib_ids.data_ptr<int>(),
        contrib_alpha.data_ptr<float>(),
        contrib_T.data_ptr<float>(),
        contrib_clamped.data_ptr<std::uint8_t>(),
        contrib_count.data_ptr<int>(),
        overflow.data_ptr<int>(),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(max_contrib));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {image, contrib_ids, contrib_alpha, contrib_T, contrib_clamped, contrib_count, overflow};
}

std::vector<torch::Tensor> composite_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor xy,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor contrib_ids,
    torch::Tensor contrib_alpha,
    torch::Tensor contrib_T,
    torch::Tensor contrib_clamped,
    torch::Tensor contrib_count,
    int64_t max_contrib) {
    const int height = static_cast<int>(grad_out.size(0));
    const int width = static_cast<int>(grad_out.size(1));
    auto g_xy = torch::zeros_like(xy);
    auto g_conic = torch::zeros_like(conic);
    auto g_opacity = torch::zeros_like(opacity);
    auto g_color = torch::zeros_like(color);
    const int threads = 256;
    const int blocks = (height * width + threads - 1) / threads;
    composite_backward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out.data_ptr<float>(),
        xy.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        contrib_ids.data_ptr<int>(),
        contrib_alpha.data_ptr<float>(),
        contrib_T.data_ptr<float>(),
        contrib_clamped.data_ptr<std::uint8_t>(),
        contrib_count.data_ptr<int>(),
        color.data_ptr<float>(),
        g_xy.data_ptr<float>(),
        g_conic.data_ptr<float>(),
        g_opacity.data_ptr<float>(),
        g_color.data_ptr<float>(),
        height,
        width,
        static_cast<int>(max_contrib));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_xy, g_conic, g_opacity, g_color};
}
