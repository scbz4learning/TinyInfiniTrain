# TinyInfiniTrain 作业报告

为了方便整理学习过程, Infinitensor的学习笔记都单独建了一个repo, [本科目的完整学习过程在这里](https://github.com/scbz4learning/Infinitensor/tree/main/docs/Stage_1/2_TinyInfiniTrain.md).
---

## 一、test 通过截图

## 二、作业步骤

> 将代码填入下面代码块中指定位置，并详细描述完成该作业的解决思路和遇到的问题。

### 作业一：autograd机制调用Neg kernel的实现

难度：⭐

对应测例：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`

需要实现的代码块位置：`infini_train/src/autograd/elementwise.cc`

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属kernel，对输入张量进行取反操作
    // HINT: 依赖test_dispatcher，kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(input_tensors.size(), 1); // 取反操作只需要一个对象
    const auto &input = input_tensors[0];

    // there is no need for previous grad. The \partial (-x) = x, which means we need current grad only.
    // Might be the only special case.
    // // No function to setup context in the class
    // saved_tensors_ = {input};

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属的反向传播kernel，计算梯度
    // HINT: 依赖test_dispatcher，kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    // return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input)};
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路

仿照别的算子写。写的过程中，观察怎么获取device和kernel的。

#### 遇到问题
`Neg` 的反向传播不需要 `grad_output`, 因为 $(-x)' = -1$.  


### 作业二：实现矩阵乘法

难度：⭐⭐

#### CPU实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`, `TEST(MatmulTest, BackwardPass)`

需要实现的代码块位置：`infini_train/src/kernels/cpu/linear.cc`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================

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
        for (int b = 0; b < bs; b++) {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                A(reinterpret_cast<float *>(input->DataPtr()) + b * M * N, M, N);

            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                B(reinterpret_cast<float *>(other->DataPtr()) + b * N * K, N, K);

            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                C(reinterpret_cast<float *>(output->DataPtr()) + b * M * K, M, K);

            C.noalias() = A * B; 
        }
        return output;
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
            
        const auto &input_dims = input->Dims();
        const auto input_dim_size = input_dims.size();
        CHECK_GE(input_dim_size, 2);

        const auto M = *(input_dims.rbegin() + 1);
        const auto N = *(input_dims.rbegin());

        const auto &other_dims = other->Dims();
        const auto other_dim_size = other_dims.size();
        CHECK_GE(other_dims.size(), 2);

        CHECK_EQ(N, *(other_dims.rbegin() + 1));
        const auto K = *(other_dims.rbegin());

        // suppose no broadcast
        const auto bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});

        auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
        auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);

        for (auto b = 0; b < bs; b++){
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                A(reinterpret_cast<float *>(input->DataPtr()) + b * M * N, M, N);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                grad_A(reinterpret_cast<float *>(grad_input->DataPtr()) + b * M * N, M, N);

            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                B(reinterpret_cast<float *>(other->DataPtr()) + b * N * K, N, K);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                grad_B(reinterpret_cast<float *>(grad_other->DataPtr()) + b * N * K, N, K);

            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                grad_C(reinterpret_cast<float *>(grad_output->DataPtr()) + b * M * K, M, K);

            grad_A.noalias() = grad_C * B.transpose();
            grad_B.noalias() = A.transpose() * grad_C; 
        }
        return {grad_input, grad_other};
    }
```

#### CUDA实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`,`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`,`TEST(MatmulTest, BackwardPassCuda)`

需要实现的代码块位置：`infini_train/src/kernels/cuda/linear.cu`

```c++
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

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, K, N,
        &alpha,
        A, M, strideA,
        B, N, strideB,
        &beta,
        C, M, strideC,
        bs));

    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

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

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        M, N, K,
        &alpha,
        grad_C, M, strideC,
        B, N, strideB,
        &beta,
        grad_A, M, strideA,
        bs));
    
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, K, M,
        &alpha,
        A, M, strideA,
        grad_C, M, strideC,
        &beta,
        grad_B, N, strideB,
        bs));

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_other};
}
```

#### 解决思路
CPU 部分：参考 `LinearForward` 和 `LinerBackward` 的实现写.  
GPU 部分：参考 CPU 写。
想到 cuda 编程说的，不要用 for 循环，在 host 和 GPU 之前传多次。所以借助了 GPT 换成了 batched 版本

#### 遇到问题
CPU 部分, batch size 是要单独拿出来的. `Eigen` 并不支持高维矩阵的直接计算, 因为会折叠成向量, 导致 \[bs1, M, N\] 和 \[bs2, N, K\] 中, N 对不齐. 解决的办法是先 map. 数据指针在 class Tensor 的 cc 文件可以找到定义. 但是最后也不知道怎么找到 `Map`. 最后的解决办法是 GPT. 

GPU 的 cublas 是列优先的。感觉之前学过，矩阵乘法，ikj 顺序比 ijk 的 locality 好很多。估计是这个原因？

### 作业三：实现Adam优化器

难度：⭐

#### CPU实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdate)`,`TEST(AdamOptimizerTest, MomentumAccumulation)`

代码位置：infini_train/src/kernels/cpu/accumulate_grad.cc

```c++
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

    m->EigenMatrix() = beta1 * m->EigenMatrix() + (1-beta1) * grad->EigenMatrix();
    // 这里要逐元素平方，为什么？Adam优化那里，v=...g^2 不是矩阵乘，而是矩阵逐元素乘法吗？
    v->EigenMatrix() = beta2 * v->EigenMatrix() + (1-beta2) * grad->EigenMatrix().array().square().matrix(); // x: grad->E * grad->E

    // 这里不用可以省略 m_hat, v_hat 表达式，因为Eigen会延迟计算，不会复制一份过来
    // 同时，保持array不用来回转
    auto m_hat = m->EigenMatrix().array() / (1 - std::pow(beta1, t));
    auto v_hat = v->EigenMatrix().array() / (1 - std::pow(beta2, t));

    // param->EigenMatrix() -= learning_rate * (m->EigenMatrix / (1 - std::pow(beta1, t))) / ((v->EigenMatrix() / (1 - pow(beta2, t))).array().sqrt() + eps);
    param->EigenMatrix().array() -= learning_rate * m_hat / (v_hat.sqrt() + eps);
}
```

#### CUDA实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`,`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`

代码位置：infini_train/src/kernels/cuda/accumulate_grad.cu

```c++
__global__ void AdamAccumulateGradKernel(float* param_ptr,
                                        const float* m_ptr,
                                        const float* v_ptr,
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

    // 计算偏执修正系数，避免GPU重复计算
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
```

#### 解决思路
```
// 这里要逐元素平方，为什么？Adam优化那里，v=...g^2 不是矩阵乘，而是矩阵逐元素乘法吗？
v->EigenMatrix() = beta2 * v->EigenMatrix() + (1-beta2) * grad->EigenMatrix().array().square().matrix(); // x: grad->E * grad->E
```
- [x] GPT 说就是逐元素平方

之后就是套公式就好

#### 遇到问题


### 作业四：实现Tensor基础操作

#### 实现Tensor的Flatten操作

难度：⭐

对应测例：`TEST(TensorTransformTest, Flatten2DTo1D)`,`TEST(TensorTransformTest, FlattenWithRange) `,`TEST(TensorTransformTest, FlattenNonContiguous)`

代码位置：infini_train/src/tensor.cc

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    // return Contiguous()->View(new_shape);
    // =================================== 作业 ===================================
    // TODO：实现张量扁平化操作，将指定维度范围[start, end]内的所有维度合并为一个维度
    // HINT:
    // =================================== 作业 ===================================

    // support neg dim
    if (start < 0) {
        start += dims_.size();
    }

    if (end < 0) {
        end += dims_.size();
    }

    CHECK_GE(start, 0);
    CHECK_GE(end, start);
    CHECK_LT(end, dims_.size());
    
    const auto flattened_dim = std::accumulate(dims_.begin() + start, dims_.begin() + end + 1, 1, std::multiplies<int64_t>{});
    auto new_shape = dims_;
    new_shape[end] = flattened_dim;
    new_shape.erase(new_shape.begin() + start, new_shape.begin() + end);

    return Contiguous()->View(new_shape);
}
```

#### 实现Tensor的反向传播机制

难度：⭐

对应测例：`TEST(TensorAutogradTest, BackwardComputesGradient)`,`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`

代码位置：infini_train/src/tensor.cc

```c++
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    // =================================== 作业 ===================================
    // TODO：实现自动微分反向传播
    // 功能描述：1. 计算当前张量对叶子节点的梯度    2. 支持多输出场景的梯度累加
    // =================================== 作业 ===================================
    // 叶子节点的建立AccumulateGrad类的过程也在BackwardPartial里面了
    CHECK(requires_grad_);
    if (!gradient) {
        // gradient 是可以为空的，默认值是nullptr。定义在.h文件。不造一个gradient会segfault
        // 要造一个全1的Tensor类
        // 根据torch语法想到Ones，搜索到nn functional里有Ones
        // 但是用init的更好，因为头文件里有了
        // function里的也是调用的init
        auto ones = std::make_shared<Tensor>(dims_, DataType::kFLOAT32);
        gradient = infini_train::nn::init::Ones(ones);
    }
    grad_fn_->BackwardPartial(gradient, output_idx_);
}
```

#### 解决思路
第一个明显不涉及数据，只是组织变一下。看到上面有个函数使用了 `Contiguous()->View(new_shape)`，然后查了一下知道这是啥意思就好做了。
第二个主要就是想到 Ones 就好了。

#### 遇到问题



### 作业五 注册算子kernel的实现

难度：⭐⭐⭐

对应测例：`TEST(DispatcherTest, RegisterAndGetKernel)`,`TEST(DispatcherTest, DuplicateRegistration)`,`TEST(DispatcherTest, GetNonexistentKernel)`

代码位置：infini_train/include/dispatcher.h

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    // =================================== 作业 ===================================
    // TODO：实现通用kernel调用接口
    // 功能描述：将存储的函数指针转换为指定类型并调用
    // =================================== 作业 ===================================

    using FuncT = RetT (*)(ArgsT...);
    // TODO: 实现函数调用逻辑

    FuncT func = reinterpret_cast<FuncT>(func_ptr_);
    return func(std::forward<ArgsT>(args)...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    // =================================== 作业 ===================================
    // TODO：实现kernel注册机制
    // 功能描述：将kernel函数与设备类型、名称绑定
    // =================================== 作业 ===================================

    // without this, there is no warning when adding an existed kernel
    // however, what is the problem? 
    // It seems that re-register multiple times should not affect much?
    CHECK(!key_to_kernel_map_.contains(key)) 
        << "Kernel already registered: " << key.second;
    key_to_kernel_map_.emplace(key, KernelFunction(kernel));
}

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static bool registery_##kernel_name##_##__LINE__ = []() {                                                            \
        infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                          \
        return true;                                                                                                   \
    }();
    // =================================== 作业 ===================================
    // TODO：实现自动注册宏
    // 功能描述：在全局静态区注册kernel，避免显式初始化代码
    // =================================== 作业 ===================================
```

#### 解决思路
1. 我原本是直接 `return` 做的，能过test。然后后面两个不会做，就去看了原版 InfiniTrain 项目，发现有这个函数，然后 GPT 一下说更安全，就换成这样了。  
2. 注册很容易。为什么要判重没理解  
3. 根本不知道怎么注册宏。。。去看了原版的 InfiniTrain 项目，Register 是 bool 类型的，直接 static 定义静态变量就行了。这里 void 没有返回值，想不到怎么做。还是 GPT。（lambda函数，拼接变量名，字符串化操作，啥的都是在这里新学的）  

还有个有意思的地方是，宏内部不能单行注释，原理是: `\` 会拼接行，所以后面一行会进注释。

#### 遇到问题
注册的时候  
1. 逻辑上：为啥要看是不是已经存在 key 了？如果重复直接替换好像感觉也不错？
2. 实现上：不加判重，test 会报错。但是具体为什么？


### 作业六：实现GPT-2整体训练

难度：⭐⭐⭐⭐

对应测例：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 训练过程logits对比

完成以上所有作业，补齐训练框架的所有实现，理论上`TEST_F(GPT2TrainingTest, LogitsConsistency)`可以通过，在用例中判断比较预置的值和单步正向传播计算结果是否在误差允许范围内相等。

#### 数据读取实现

代码位置：example/common/tiny_shakespeare_dataset.cc

```c++
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */

    std::ifstream fin(path, std::ios::binary);
    CHECK(fin) << "Failed to open file" << path;

    // magic
    // the magic number is used for identify dataset
    // and to specify the kTypeMap
    // how do I know????? (I cheated by looking others' implementation)
    // From https://github.com/InfiniTensor/InfiniTrain/blob/master/example/common/tiny_shakespeare_dataset.cc
    TinyShakespeareFile text_file;
    const auto header = ReadSeveralBytesFromIfstream(1024, &fin);
    const int32_t magic = BytesToType<int32_t>(header, 0); // sec param is offset from beginning
    const int32_t version = BytesToType<int32_t>(header, 4);
    const int32_t num_tokens = BytesToType<int32_t>(header, 8);
    text_file.type = kTypeMap.at(magic);

    // There are tokens of size `num_tokens`
    // The tokens are divided into sequence, whose size is fixed as `sequence_length`
    // Therefore, there are `num_sequences` sequences
    const int num_sequences = num_tokens / sequence_length;
    // vector assign: replace the content with
    // 1. count & values
    // 2. from first to last
    // 3. init list
    // Each sequence is mapped to a tensor,
    //      therefore, the dim is of size `num_sequences` and each dim is of same size `sequence_length`
    // The signature of `dims` is `std::vector<int64_t>`, so it is nec to map before use
    // x: There is no padding processing!
    // Actually there is, the last sequence will be dropped if not complete
    text_file.dims.assign(num_sequences, static_cast<int64_t>(sequence_length));

    const int data_size_in_bytes
        = std::accumulate(text_file.dims.begin(),
                        text_file.dims.end(),
                        kTypeToSize.at(text_file.type), // change the init value here
                         std::multiplies<int>{}); // use {} instead
    // Init tensor, type is kINIT64? why?
    text_file.tensor = infini_train::Tensor(text_file.dims, DataType::kINT64);
    // let data_ptr_ pointing to the data 
    int64_t *dst = static_cast<int64_t *>(text_file.tensor.DataPtr());

    // // 安全的 union，这里要么存 std::vector<uint16_t>, 要么存 std::vector<int32_t>
    // std::variant<std::vector<uint16_t>, std::vector<int32_t>> buffer;
    // if (text_file.type == TinyShakespeareType::kUINT16) {
    //     CHECK_LE(sequence_length, 1024); // GPT-2: max_seq_length = 1024
    //     buffer = std::vector<uint16_t>(num_sequences * sequence_length);
    // } else if (text_file.type == TinyShakespeareType::kUINT32) {
    //     CHECK_LE(sequence_length, 8192); // LLaMA-3: max_seq_length = 8192
    //     buffer = std::vector<int32_t>(num_sequences * sequence_length);
    // }

    // // 用 visit 来访问并 cast。
    // std::visit(
    //     // 捕获外部变量，通过 引用
    //     // [=] 就是捕获 值
    //     [&](auto &vec) {
    //         fin.read(reinterpret_cast<char *>(vec.data()), data_size_in_bytes);
    //         for (size_t i = 0; i < vec.size(); ++i) { dst[i] = static_cast<int64_t>(vec[i]); }
    //     },
    //     buffer);

    // 感觉上面很复杂。。。 明明都 if 了， 不如直接 cast
    if (text_file.type == TinyShakespeareType::kUINT16) {
        std::vector<uint16_t> vec(num_sequences * sequence_length);
        fin.read(reinterpret_cast<char *>(vec.data()), data_size_in_bytes);
        for (size_t i = 0; i < vec.size(); ++i) { dst[i] = static_cast<int64_t>(vec[i]); }
    } else if (text_file.type == TinyShakespeareType::kUINT32) {
        std::vector<int32_t> vec(num_sequences * sequence_length);
        fin.read(reinterpret_cast<char *>(vec.data()), data_size_in_bytes);
        for (size_t i = 0; i < vec.size(); ++i) { dst[i] = static_cast<int64_t>(vec[i]); }
    }
    return text_file;
}


// 用 infinitrain 的 写法，直接成员初始化
TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)), num_samples_(text_file_.dims[0] - 1) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    // 用 infinitrain 的 写法，直接成员初始化
    CHECK_EQ(text_file_.dims[1], sequence_length_);
    CHECK_EQ(static_cast<int>(text_file_.tensor.Dtype()), static_cast<int>(DataType::kINT64));
}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    std::ifstream fin(filepath, std::ios::binary);
    CHECK(fin) << "Failed to open file" << filepath;

    const auto header = ReadSeveralBytesFromIfstream(1024, &fin);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    const uint32_t version = BytesToType<uint32_t>(header, 4);
    vocab_size_ = BytesToType<uint32_t>(header, 8);
    eot_token_ = kEotMap.at(magic_number_);
    
    for (uint32_t i = 0; i < vocab_size_; i++) {
        auto curSizeByte = ReadSeveralBytesFromIfstream(2, &fin);
        size_t curSize = BytesToType<size_t>(curSizeByte, 0);
        auto tokenByte = ReadSeveralBytesFromIfstream(curSize, &fin);
        // memcpy 不能作用于变长类型
        // std::string token = BytesToType<std::string>(tokenByte, 0);
        std::string token(tokenByte.begin(), tokenByte.end());
        token_table_.push_back(token);
    }
}
```

```c++
std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    CHECK_LT(token_id, token_table_.size());
    return token_table_[token_id];
}
```

```c++
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    /* ...原代码... */
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
    }
    std::cout << std::endl;
}
```

#### 解决思路
##### Dataset
一开始做作业的时候没看到头文件的注释 `https://github.com/karpathy/llm.c/blob/master/dev/data/tinyshakespeare.py`，完全看不懂应该怎么实现。GPT说这个也不是标准格式。所以只能去找 InfiniTrain 原项目。头文件的字段，哪个有用哪个没用，magic 字段对应 GPT-2 或者 LLaMA 3 是真想不到。。。我一开始以为头全都丢，然后使用的时候自行调用，类似于传命令行参数来决定用什么模型。

所以 dataset 基本就是抄完了，然后读懂了。

##### Tokenizer
Tokenizer 的部分更混乱。。。首先有了上面的dataset的经验，而且class里有成员，所以头怎么读是比较清楚的。

但是词表格式不知道怎么看。  

```shell
$ ls -R Data/
Data/:
gpt2_124M.bin  gpt2_124M_bf16.bin  gpt2_124M_debug_state.bin  gpt2_logits_reference.bin  gpt2_tokenizer.bin  tinyshakespeare

Data/tinyshakespeare:
tiny_shakespeare_train.bin  tiny_shakespeare_val.bin
```

借助gpt，先搞明白文件目录。需要的文件是  
1. gpt2_logits_reference.bin：gpt的词表  
2. tiny_shakespeare_train.bin：之前dataset函数需要的整数二进制文件  

所以 dataset 是里有整数序列，可以直接读，但是词表格式没法知道。这时候`xxd`一下：

```xxd
00000400: 0121 0122 0123 0124 0125 0126 0127 0128  .!.".#.$.%.&.'.(
00000410: 0129 012a 012b 012c 012d 012e 012f 0130  .).*.+.,.-.../.0
00000420: 0131 0132 0133 0134 0135 0136 0137 0138  .1.2.3.4.5.6.7.8
00000430: 0139 013a 013b 013c 013d 013e 013f 0140  .9.:.;.<.=.>.?.@
00000440: 0141 0142 0143 0144 0145 0146 0147 0148  .A.B.C.D.E.F.G.H
00000450: 0149 014a 014b 014c 014d 014e 014f 0150  .I.J.K.L.M.N.O.P
00000460: 0151 0152 0153 0154 0155 0156 0157 0158  .Q.R.S.T.U.V.W.X
00000470: 0159 015a 015b 015c 015d 015e 015f 0160  .Y.Z.[.\.].^._.`
00000480: 0161 0162 0163 0164 0165 0166 0167 0168  .a.b.c.d.e.f.g.h
00000490: 0169 016a 016b 016c 016d 016e 016f 0170  .i.j.k.l.m.n.o.p
000004a0: 0171 0172 0173 0174 0175 0176 0177 0178  .q.r.s.t.u.v.w.x
000004b0: 0179 017a 017b 017c 017d 017e 01a1 01a2  .y.z.{.|.}.~....
...
00000600: 0220 7402 2061 0268 6502 696e 0272 6502  . t. a.he.in.re.
00000610: 6f6e 0420 7468 6502 6572 0220 7302 6174  on. the.er. s.at
00000620: 0220 7702 206f 0265 6e02 2063 0269 7402  . w. o.en. c.it.
00000630: 6973 0261 6e02 6f72 0265 7302 2062 0265  is.an.or.es. b.e
00000640: 6402 2066 0369 6e67 0220 7002 6f75 0320  d. f.ing. p.ou. 
00000650: 616e 0261 6c02 6172 0320 746f 0220 6d03  an.al.ar. to. m.
00000660: 206f 6603 2069 6e02 2064 0220 6804 2061   of. in. d. h. a
00000670: 6e64 0269 6302 6173 026c 6503 2074 6803  nd.ic.as.le. th.
```

借助 GPT，词表格式有定长和变长，这里很明显的是，`0x400` 附近像是符号和字母表，字母之间的间隔不多，`0x600` 附近开始像是有词了，但是看起来有offset。这是后就挺明确是变长的了。借助GPT，`0x400` 开始 `0121` 一个字节的长度，`21` 确实对应`!`。所以读序列的时候，读一个 `kUINT16` 的长度，这是一个词的长度，然后按照值读string并cast存类里面就好了。

具体的实现还是挺简单的，毕竟辅助函数都给写好了。。。

##### 文本生成


#### 遇到问题

