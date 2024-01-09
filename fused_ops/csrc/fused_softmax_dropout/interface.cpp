#include <torch/extension.h>
#include <ATen/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <vector>

std::vector<c10::optional<torch::Tensor>> fwd_cuda(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& input, 
                               float                dropout_prob,
                               c10::optional<at::Generator> gen_
                                                  );

torch::Tensor bwd_cuda(
                        int heads,
                        torch::Tensor const& output_grads,
                        torch::Tensor const& softmax_results,
                        c10::optional<torch::Tensor> const& dropout_mask,
                        float                dropout_prob
                                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<c10::optional<torch::Tensor>> fwd(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& input,
                               float                dropout_prob,
                               c10::optional<at::Generator> gen_
                                                 )
{
  AT_ASSERTM(input.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 || input.scalar_type() == at::ScalarType::Float, "Only HALF/BFloat16/Float is supported");


  return fwd_cuda(
                                 is_training,
                                 heads, 
                                 input, 
                                 dropout_prob,
                                 gen_
                                );
}

torch::Tensor bwd(int heads,
                torch::Tensor const& output_grads, 
                torch::Tensor const& softmax_results,
                c10::optional<torch::Tensor> const& dropout_mask,
                float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(!dropout_mask || dropout_mask->dim()      == 1, "expected 1D tensor");

  AT_ASSERTM(output_grads.scalar_type()      == at::ScalarType::Half || output_grads.scalar_type()      == at::ScalarType::BFloat16 || output_grads.scalar_type()      == at::ScalarType::Float, "Only HALF/BFloat16/Float is supported");
  AT_ASSERTM(softmax_results.scalar_type()   == at::ScalarType::Half || softmax_results.scalar_type()   == at::ScalarType::BFloat16 || softmax_results.scalar_type()   == at::ScalarType::Float, "Only HALF/BFloat16/Float is supported");

  return bwd_cuda(
                heads,
                output_grads,
                softmax_results, 
                dropout_mask,
                dropout_prob
                );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwd, "softmax dropout -- Forward.");
  m.def("backward", &bwd, "softmax dropout -- Backward.");
}