#include "cublas_v2.h"
#include "glog/logging.h"
#include <cub/block/block_reduce.cuh>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

// 每次调用 cublas 肯定效率很低。GPT 给了 `cublasSgemmStridedBatched` 函数
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    // Check input is matrices
    // Check input is matrices
    const auto &input_dims = input->Dims();
    const auto input_dim_size = input_dims.size();
    CHECK_GE(input_dim_size, 2);

    const auto M = *(input_dims.rbegin() + 1);
    const auto N = *(input_dims.rbegin());

    // x: ther can be scalar
    // no it cannot
    const auto &other_dims = other->Dims();
    const auto other_dim_size = other_dims.size();
    CHECK_GE(other_dims.size(), 2);

    CHECK_EQ(N, *(other_dims.rbegin() + 1));
    const auto K = *(other_dims.rbegin());

    // suppose no broadcast
    auto output_dims = input_dims;
    output_dims.back() = other_dims.back();
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    const auto bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float* A = reinterpret_cast<const float *>(input->DataPtr());
    const float* B = reinterpret_cast<const float *>(other->DataPtr());
    float* C = reinterpret_cast<float *>(output->DataPtr());

    // stride
    const long long strideA = M * N;
    const long long strideB = N * K;
    const long long strideC = M * K;

    // cuBLAS 会先跳着读，相当于先复制一份改变顺序的矩阵，再1比1传给GPU
    //  （实际当然不需要，只需要换组织方式就行）
    // 所以 (2,3,4) 的矩阵
    // [[a11, a12, a13, a14
    //   a21, a22, a23, a24,
    //   a31, a32, a33, a34],
    
    //  [b11, b12, b13, b14
    //   b21, b22, b23, b24,
    //   b31, b32, b33, b34]]
    // 
    // 内存中应该是
    // a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, b11, b12, b13, b14 ...
    // 
    // cuBLAS 会复制成
    // a11, a21, a31, a12, ...
    //
    // 然后重排成
    // 所以 (2,3,4) 的矩阵
    // [[a11, a21, a31
    //   a12, a22, a13,
    //   a13, a23, a33,
    //   a14, a24, a34],
    
    //  [b11, b21, b31
    //   b12, b22, b13,
    //   b13, b23, b33,
    //   b14, b24, b34]]
    // 
    // 现在每一行是一个向量了，每个向量中包含3个元素，所以主序是3
    //  对的！主序不是向量的个数，而是向量中包含元素的个数，即向量大小
    //  所以一个矩阵 (bs, m, n) 如果什么都不做，
    //      会被解释成 (bs, n, m)，主序m，主序是行优先的m
    //
    // 所以为了计算
    // A * B -> C
    // (bs, M, N) * (bs, N, K) -> (bs, M, K)
    // 
    // 也就是得到内存的 C (bs, M, K)
    // 需要 cuBLAS的 (bs, K, M) 主序M
    //
    // 所以要计算
    // B^T * A^T
    //
    // B什么也不做就会被解释成(bs, K, N)，主序N
    // A什么也不做就会被解释成(bs, N, M)，主序M
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        K, M, N, // cuBLAS布局得到的 m n 公共
        &alpha,
        B, N, strideB,
        A, M, strideA,
        &beta,
        C, M, strideC,
        bs));

    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

// 每次调用 cublas 肯定效率很低。GPT 给了 `cublasSgemmStridedBatched` 函数
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    // Check input is matrices
    const auto &input_dims = input->Dims();
    const auto input_dim_size = input_dims.size();
    CHECK_GE(input_dim_size, 2);

    const auto M = *(input_dims.rbegin() + 1);
    const auto N = *(input_dims.rbegin());

    // suppose no broadcast
    const auto &other_dims = other->Dims();
    const auto other_dim_size = other_dims.size();
    CHECK_GE(other_dims.size(), 2);

    CHECK_EQ(N, *(other_dims.rbegin() + 1));
    const auto K = *(other_dims.rbegin());

    const auto bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float* A = reinterpret_cast<const float *>(input->DataPtr());
    const float* B = reinterpret_cast<const float *>(other->DataPtr());
    const float* grad_C = reinterpret_cast<const float *>(grad_output->DataPtr());
    float* grad_A = reinterpret_cast<float *>(grad_input->DataPtr());
    float* grad_B = reinterpret_cast<float *>(grad_other->DataPtr());

    // Strides
    const long long strideA = M * N;
    const long long strideB = N * K;
    const long long strideC = M * K;

    // grad_A = grad_C * T(B)
    // grad_C [bs, M, K]
    // B      [bs, N, K]
    // grad_A [bs, M, N]
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        grad_C, K, strideC,
        B, K, strideB,
        &beta,
        grad_A, N, strideA,
        bs));
    
    // grad_B = T(A) * grad_C
    // A      [bs, M, N]
    // grad_C [bs, M, K]
    // grad_B [bs, N, K]
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        K, N, M,
        &alpha,
        A, N, strideA,
        grad_C, K, strideC,
        &beta,
        grad_B, K, strideB,
        bs));

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_other};
}

__global__ void BiasCopyKernel(float *output, const float *bias, int bs, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * out_features) {
        return;
    }
    int j = idx % out_features;
    output[idx] = bias[j];
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {

    /*
        !transpose: output = input * weight + bias
        output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

        transpose:  output = input * weight^T + bias
        output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);

    // As for cublas:
    // C = alpha * op(B) * op(A) + beta * C
    // Dimensions:
    //   input:  (bs, in_features)
    //   weight: (in_features, out_features) or (out_features, in_features) if transposed
    //   output: (bs, out_features)
    const int64_t out_features = weight_dims[transpose ? 0 : 1];

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;
        BiasCopyKernel<<<num_blocks, threads_per_block>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), bs, out_features);
    } else {
        output->Fill<float>(0.0f);
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (transpose) {
        // weight is [out_features, in_features] here

        // output = input * weight.T --> output.T = weight * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[in_features, out_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    } else {
        // output = input * weight --> output.T =  weight.T * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[out_features, in_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

template <int BLOCK_SIZE>
__global__ void ReduceColumnsKernel(const float *__restrict__ input, float *__restrict__ output, int num_rows,
                                    int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    float sum = 0.0f;

    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) { sum += input[row * num_cols + col]; }

    float reduced = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        output[row] = reduced;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32,
                                             grad_output->GetDevice());
        grad_bias->Fill<float>(0.0f);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (transpose) {
        // weight is [out_features, in_features] here

        // d_input = d_output * weight --> d_input.T = weight.T * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[in_features, out_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
        // C = d_weight.T[in_features, out_features]
        // A = input.T[in_features, bs]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, bs, &alpha,
                                 static_cast<const float *>(input->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), in_features));
    } else {
        // weight is [in_features, out_features] here

        // d_input = d_output * weight.T --> d_input.T = weight * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[out_features, in_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
        // C = d_weight.T[out_features, in_features]
        // A = d_output.T[out_features, bs]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, bs, &alpha,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), out_features));
    }

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    if (bias) {
        constexpr int BLOCK_SIZE = 256;
        int threads_per_block = BLOCK_SIZE;
        int num_blocks = out_features;
        ReduceColumnsKernel<BLOCK_SIZE>
            <<<num_blocks, threads_per_block>>>(static_cast<const float *>(grad_output->DataPtr()),
                                                static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CUDA_LINEAR_KERNEL
