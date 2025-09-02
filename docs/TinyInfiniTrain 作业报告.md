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
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
    }
```

#### 解决思路
CPU 部分 参考 `LinearForward` 和 `LinerBackward` 的实现写.  


#### 遇到问题
CPU 部分, batch size 是要单独拿出来的. `Eigen` 并不支持高维矩阵的直接计算, 因为会折叠成向量, 导致 \[bs1, M, N\] 和 \[bs2, N, K\] 中, N 对不齐. 解决的办法是先 map. 数据指针在 class Tensor 的 cc 文件可以找到定义. 但是最后也不知道怎么找到 `Map`. 最后的解决办法是 GPT. 


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
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF: 
    // =================================== 作业 ===================================
}
```

#### 解决思路



#### 遇到问题
```
// 这里要逐元素平方，为什么？Adam优化那里，v=...g^2 不是矩阵乘，而是矩阵逐元素乘法吗？
v->EigenMatrix() = beta2 * v->EigenMatrix() + (1-beta2) * grad->EigenMatrix().array().square().matrix(); // x: grad->E * grad->E
```

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
}

#define REGISTER_KERNEL(device, kernel_name, kernel_func) \
    // =================================== 作业 ===================================
    // TODO：实现自动注册宏
    // 功能描述：在全局静态区注册kernel，避免显式初始化代码
    // =================================== 作业 ===================================
    CHECK(!key_to_kernel_map_.contains(key)) 
        << "Kernel already registered: " << key.second;
    key_to_kernel_map_.emplace(key, KernelFunction(kernel));
```

#### 解决思路



#### 遇到问题



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
}
```

```c++
std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
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

一开始做作业的时候没看到头文件的注释 `https://github.com/karpathy/llm.c/blob/master/dev/data/tinyshakespeare.py`，完全看不懂应该怎么实现。GPT说这个也不是标准格式。所以只能去找 InfiniTrain 原项目。头文件的字段，哪个有用哪个没用，magic 字段对应 GPT-2 或者 LLaMA 3 是真想不到。。。我一开始以为头全都丢，然后使用的时候自行调用，类似于传命令行参数来决定用什么模型。

所以 dataset 基本就是抄完了，然后读懂了。



#### 遇到问题

