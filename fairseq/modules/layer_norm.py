# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import fused_ops
    import fused_layer_norm_fast_cuda
    import fused_layer_norm_backward_gamma_beta_cuda

    has_fused_layer_norm_fast = True


    class FusedLayerNormFastFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, normalized_shape, eps, is_training):
            ctx.normalized_shape = normalized_shape
            ctx.eps = eps
            input_ = input.contiguous()
            weight_ = weight
            bias_ = bias
            output, mean, invvar = fused_layer_norm_fast_cuda.forward(
                input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
            if is_training:
                ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input_, weight_, bias_, mean, invvar = ctx.saved_tensors
            grad_output = grad_output.contiguous()
            grad_input = fused_layer_norm_fast_cuda.backward(
                grad_output, mean, invvar,
                input_, ctx.normalized_shape,
                weight_, bias_, ctx.eps)
            grad_weight, grad_bias = fused_layer_norm_backward_gamma_beta_cuda.backward_gamma_beta(
                grad_output, mean, invvar,
                input_, ctx.normalized_shape,
                weight_, bias_, ctx.eps)
            return grad_input, grad_weight, grad_bias, None, None, None


    class FusedLayerNormFast(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            super(FusedLayerNormFast, self).__init__()
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = torch.Size(normalized_shape)
            self.eps = eps
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return F.layer_norm(
                    x, self.normalized_shape, self.weight, self.bias, self.eps)
            return FusedLayerNormFastFunction.apply(
                x, self.weight, self.bias, self.normalized_shape, self.eps, self.training)

        def extra_repr(self):
            return '{normalized_shape}, eps={eps}, ' \
                   'elementwise_affine=True'.format(**self.__dict__)

except ImportError:
    has_fused_layer_norm_fast = False

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layer_norm = True


    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layer_norm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    if not export and torch.cuda.is_available():
        if has_fused_layer_norm_fast and elementwise_affine:
            return FusedLayerNormFast(normalized_shape, eps)
        elif has_fused_layer_norm:
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
