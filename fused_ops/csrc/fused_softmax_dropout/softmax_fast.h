#pragma once
#include <iostream>
#include <type_traits>
#include <limits>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

#define CELL(a, b) (((a) + (b) - 1) / (b))
#if __cplusplus >= 201703L
    #define IF_CONSTEXPR constexpr
#else
    #define IF_CONSTEXPR
#endif
//#define C10_WARP_SIZE 32
//#define TORCH_INTERNAL_ASSERT(x)

template <int N>
using IntegerBits = typename std::conditional<N <= 8, uint8_t, 
    typename std::conditional<N <= 16, uint16_t,
        typename std::conditional<N <= 32, uint32_t,
            typename std::conditional<N <= 64, uint64_t, void>::type
        >::type
    >::type
>::type;

inline int log2_ceil_native(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}


// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
// mask's size: batch_size * WARP_SIZE * sizeof(MaskType)
// dst = Softmax(...) * mask / p
// dst_orig = Softmax(...)
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE, bool need_mask>
__global__ void softmax_warp_forward(input_t *dst, input_t *dst_orig, const output_t *src,
    IntegerBits<WARP_ITERATIONS> *mask, acc_t p, int batch_size, int stride, int element_count, uint64_t seed, uint64_t rand_offset) {
    using MaskType = IntegerBits<WARP_ITERATIONS>;
    curandStatePhilox4_32_10_t state;
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
    const int thread_offset = first_batch * stride + local_idx;
    constexpr int mask_stride = WARP_SIZE;
    if IF_CONSTEXPR (need_mask) {
        curand_init(seed, thread_offset, rand_offset, &state);
    }
 
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;
 
    src += thread_offset;
    dst += thread_offset;
    if IF_CONSTEXPR (need_mask) {
        dst_orig += thread_offset;
        mask += first_batch * mask_stride;
    }
 
    // load data from global memory
    input_t elements_input[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            elements_input[i][it] = -std::numeric_limits<float>::infinity();
 
            if (element_index < batch_element_count) {
                elements_input[i][it] = src[i * element_count + it * WARP_SIZE];
            }
 
        }
    }
 
    // convert input_t to acc_t
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            elements[i][it] = elements_input[i][it];
        }
    }
 
    constexpr uint32_t  FULL_MASK = 0xffffffff;
 
    // compute local max_value
 
    // take the max_value of the first element to avoid one max call
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        max_value[i] = elements[i][0];
    }
 
    #pragma unroll
    for (int it = 1;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }

    // reduction max_value
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float val[WARP_BATCH];
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            val[i] = __shfl_xor_sync(FULL_MASK, max_value[i], offset, WARP_SIZE);
        }
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
        }
    }
 
    // compute local sum
    acc_t sum[WARP_BATCH] { 0.0f };
 
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            //elements[i][it] = expf(elements[i][it] - max_value[i]);
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }
 
    // reduction sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += __shfl_xor_sync(FULL_MASK, sum[i], offset, WARP_SIZE);
        }
    }

    // store result
    if IF_CONSTEXPR (need_mask) {
        const acc_t pinv = 1.0 / p;
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            if (i >= local_batches)
                break;
            MaskType m = 0;
            if IF_CONSTEXPR (WARP_ITERATIONS == 1) {
                float rand = curand_uniform(&state);
                m = rand < p;
            } else if IF_CONSTEXPR (WARP_ITERATIONS == 2) {
                m = curand_uniform(&state) < p;
                m |= (curand_uniform(&state) < p) << 1;
            } else {
                #pragma unroll
                for (int j = 0; j < CELL(WARP_ITERATIONS, 4); ++j) {
                    float4 rand4 = curand_uniform4(&state);
                    m |= (((MaskType)(rand4.x < p)) << (j * 4))
                     | (((MaskType)(rand4.y < p)) << (j * 4 + 1))
                     | (((MaskType)(rand4.z < p)) << (j * 4 + 2))
                     | (((MaskType)(rand4.w < p)) << (j * 4 + 3));
                }
            }
            mask[i * mask_stride + local_idx] = m;
            #pragma unroll
            for (int it = 0;it < WARP_ITERATIONS; ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index < element_count) {
                    const output_t d = elements[i][it] / sum[i];
                    dst[i * element_count + it * WARP_SIZE] = (acc_t)d * ((acc_t)((m >> it) & 1) * pinv);
                    dst_orig[i * element_count + it * WARP_SIZE] = d;
                }
                else {
                    break;
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            if (i >= local_batches)
                break;
            #pragma unroll
            for (int it = 0;it < WARP_ITERATIONS; ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index < element_count) {
                    dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
                }
                else {
                    break;
                }
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE>
__device__ __forceinline__ void warp_reduce_sum(acc_t* sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
            sum[i] = sum[i] + b;
        }
    }
}

inline at::ScalarType softmax_mask_dtype(int elements) {
    if (elements > 1024) {
        return torch::kInt64;
    } else if (elements > 512) {
        return torch::kInt32;
    } else if (elements > 256) {
        return torch::kInt16;
    }
    return torch::kInt8;
}

inline int softmax_mask_size(int batch_size, int elements) {
    int log2_elements = log2_ceil_native(elements);
    int e = 1 << log2_elements;
    int warp_size = e < 32 ? e : 32;
    return batch_size * warp_size;
}

inline int softmax_rng_offset(int elements) {
    int log2_elements = log2_ceil_native(elements);
    int e = 1 << log2_elements;
    int warp_iterations = e <= 32 ? 1 : e / 32;
    int warp_size = e <= 32 ? e : 32;
    int warp_batch = e <= 128 ? 2 : 1;
    return warp_iterations * warp_batch;
}

template<typename input_t, typename output_t, typename acc_t, bool need_mask>
bool dispatch_softmax_forward(output_t *dst, output_t *dst_orig, const input_t *src, void *mask, acc_t p,
    int softmax_elements, int softmax_elements_stride, int batch_count, uint64_t seed, uint64_t offset)
{
    TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 2048 );
    if (softmax_elements == 0) {
       return false;
    } else {
        int log2_elements = log2_ceil_native(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0: // 1
                softmax_warp_forward<input_t, output_t, acc_t, 2,1,1, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 1: // 2
                softmax_warp_forward<input_t, output_t, acc_t, 2,1,2, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 2: // 4
                softmax_warp_forward<input_t, output_t, acc_t, 2,1,4, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 3: // 8
                softmax_warp_forward<input_t, output_t, acc_t, 2,1,8, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 4: // 16
                softmax_warp_forward<input_t, output_t, acc_t, 2,1,16, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 5: // 32
                softmax_warp_forward<input_t, output_t, acc_t, 2,1,32, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 6: // 64
                softmax_warp_forward<input_t, output_t, acc_t, 2,2,32, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 7: // 128
                softmax_warp_forward<input_t, output_t, acc_t, 2,4,32, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 8: // 256
                softmax_warp_forward<input_t, output_t, acc_t, 1,8,32, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 9: // 512
                softmax_warp_forward<input_t, output_t, acc_t, 1,16,32, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint16_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 10: // 1024
                softmax_warp_forward<input_t, output_t, acc_t, 1,32,32, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint32_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            case 11: // 2048
                softmax_warp_forward<input_t, output_t, acc_t, 1,64,32, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, dst_orig, src,
                        (uint64_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements, seed, offset);
                return true;
            default:
                return false;
        }
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp softmax backward functions as fused variants of at::softmax_backward_data function
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE, bool is_log_softmax, bool need_mask>
__global__ void softmax_warp_backward(output_t *gradInput, const input_t *grad, const input_t *output,
    const IntegerBits<WARP_ITERATIONS> *mask, acc_t p, int batch_size, int stride, int element_count)
{
    using MaskType = IntegerBits<WARP_ITERATIONS>;
    constexpr int mask_stride = WARP_SIZE;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    // the first element to process by the current thread
    int thread_offset = first_batch * stride + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;
    if IF_CONSTEXPR (need_mask) {
        mask += first_batch * mask_stride;
    }

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS] ;
    if IF_CONSTEXPR (need_mask) {
        MaskType mask_reg[WARP_BATCH];
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            if (i >= local_batches)
                break;
            mask_reg[i] = mask[i * mask_stride + local_idx];
        }
        
        const acc_t pinv = 1.0 / p;
        
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            int batch_element_count = (i >= local_batches) ? 0 : element_count;
            MaskType m = mask_reg[i];
            #pragma unroll
            for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index < batch_element_count) {
                    grad_reg[i][it] = (input_t)((acc_t)((m >> it) & 1) * (acc_t)grad[i*element_count+it*WARP_SIZE] * pinv )*output[i*element_count+it*WARP_SIZE];;
                    output_reg[i][it] = output[i*element_count+it*WARP_SIZE];
                } else {
                    grad_reg[i][it] = acc_t(0);
                    output_reg[i][it] = acc_t(0);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            int batch_element_count = (i >= local_batches) ? 0 : element_count;
            #pragma unroll
            for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
                int element_index = local_idx + it * WARP_SIZE;
                if (element_index < batch_element_count) {
                    grad_reg[i][it] = grad[i*element_count+it*WARP_SIZE] * output[i*element_count+it*WARP_SIZE];;
                    output_reg[i][it] = output[i*element_count+it*WARP_SIZE];
                } else {
                    grad_reg[i][it] = acc_t(0);
                    output_reg[i][it] = acc_t(0);
                }
            }
        }
    }

    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = grad_reg[i][0]; 
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            sum[i] += grad_reg[i][it];
        }
    }
    warp_reduce_sum<acc_t, WARP_BATCH, WARP_SIZE>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                if IF_CONSTEXPR (is_log_softmax) {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
                } else {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - output_reg[i][it] * sum[i]);
                }
            }
        }
    }
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax, bool need_mask>
void dispatch_softmax_backward(output_t *grad_input, const input_t *grad, const input_t *output,
    const void *mask, acc_t p, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 2048 );
    if (softmax_elements == 0) {
       return;
    } else {
        int log2_elements = log2_ceil_native(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0: // 1
                softmax_warp_backward<input_t, output_t, acc_t, 2,1,1, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 1: // 2
                softmax_warp_backward<input_t, output_t, acc_t, 2,1,2, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 2: // 4
                softmax_warp_backward<input_t, output_t, acc_t, 2,1,4, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 3: // 8
                softmax_warp_backward<input_t, output_t, acc_t, 2,1,8, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 4: // 16
                softmax_warp_backward<input_t, output_t, acc_t, 2,1,16, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 5: // 32
                softmax_warp_backward<input_t, output_t, acc_t, 2,1,32, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 6: // 64
                softmax_warp_backward<input_t, output_t, acc_t, 2,2,32, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 7: // 128
                softmax_warp_backward<input_t, output_t, acc_t, 2,4,32, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 8: // 256
                softmax_warp_backward<input_t, output_t, acc_t, 1,8,32, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint8_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 9: // 512
                softmax_warp_backward<input_t, output_t, acc_t, 1,16,32, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint16_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 10: // 1024
                softmax_warp_backward<input_t, output_t, acc_t, 1,32,32, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint32_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 11: // 2048
                softmax_warp_backward<input_t, output_t, acc_t, 1,64,32, is_log_softmax, need_mask>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output,
                        (uint64_t *)mask, p, batch_count, softmax_elements_stride, softmax_elements);
                break;
            default:
                break;
        }
    }
}
