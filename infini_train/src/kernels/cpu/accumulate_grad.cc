#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    // 只迭代一次
    // m、v 已经给出，给的只读，不需要计算
    // 这里只是一个kernel，相当于Add
    
    // 补 check
    // 空指针检查
    CHECK(grad != nullptr);
    CHECK(param != nullptr);
    CHECK(m != nullptr);
    CHECK(v != nullptr);

    // 维度检查
    const size_t n = grad->NumElements();
    CHECK_EQ(param->NumElements(), n);
    CHECK_EQ(m->NumElements(), n);
    CHECK_EQ(v->NumElements(), n);

    // 步数检查
    CHECK_GT(t, 0) << "Adam step t must be >= 1";

    m->EigenMatrix() = beta1 * m->EigenMatrix() + (1-beta1) * grad->EigenMatrix();
    // 这里要逐元素平方，为什么？Adam优化那里，v=...g^2 不是矩阵乘，而是矩阵逐元素乘法吗？
    v->EigenMatrix() = beta2 * v->EigenMatrix() + (1-beta2) * grad->EigenMatrix().array().square().matrix(); // x: grad->E * grad->E

    // 这里可以不用省略 m_hat, v_hat 表达式，因为Eigen会延迟计算，不会复制一份过来
    // 同时，保持array不用来回转
    auto m_hat = m->EigenMatrix().array() / (1.0f - powf(beta1, t));
    auto v_hat = v->EigenMatrix().array() / (1.0f - powf(beta2, t));

    // param->EigenMatrix() -= learning_rate * (m->EigenMatrix / (1 - std::pow(beta1, t))) / ((v->EigenMatrix() / (1 - pow(beta2, t))).array().sqrt() + eps);
    param->EigenMatrix().array() -= learning_rate * m_hat / (v_hat.sqrt() + eps);
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
