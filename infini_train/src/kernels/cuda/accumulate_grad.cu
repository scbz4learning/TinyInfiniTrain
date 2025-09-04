#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

__global__ void AdamAccumulateGradKernel(float* param_ptr,
                                        float* m_ptr,
                                        float* v_ptr,
                                        const float* grad_ptr,
                                        const float beta1,
                                        const float beta2,
                                        const float bias_correction1,
                                        const float bias_correction2,
                                        const float learning_rate,
                                        const float eps,
                                        const size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float m_t = beta1 * m_ptr[idx] + (1 - beta1) * grad_ptr[idx];
        float v_t = beta2 * v_ptr[idx] + (1 - beta2) * grad_ptr[idx] * grad_ptr[idx];
        
        // 要写回m&v！
        m_ptr[idx] = m_t;
        v_ptr[idx] = v_t;
        
        float m_hat = m_t / bias_correction1;
        float v_hat = v_t / bias_correction2;
        param_ptr[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);

    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    // check
    CHECK(grad != nullptr);
    CHECK(param != nullptr);
    CHECK(m != nullptr);
    CHECK(v != nullptr);

    const size_t n = grad->NumElements();
    CHECK_EQ(param->NumElements(), n);
    CHECK_EQ(m->NumElements(), n);
    CHECK_EQ(v->NumElements(), n);

    CHECK_GT(t, 0) << "Adam step t must be >= 1";

    const float bias_correction1 = 1 - std::pow(beta1, t);
    const float bias_correction2 = 1 - std::pow(beta2, t);
    
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>> (
        static_cast<float *>(param->DataPtr()),
        static_cast<float *>(m->DataPtr()),
        static_cast<float *>(v->DataPtr()),
        static_cast<float *>(grad->DataPtr()),
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        learning_rate,
        eps,
        n
    );
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
