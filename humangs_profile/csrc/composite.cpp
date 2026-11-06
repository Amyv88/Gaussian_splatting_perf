#include <torch/extension.h>
#include <vector>

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
    int64_t max_contrib);

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
    int64_t max_contrib);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &composite_forward_cuda, "tiled gaussian composite forward");
    m.def("backward", &composite_backward_cuda, "tiled gaussian composite backward");
}
