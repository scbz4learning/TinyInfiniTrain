#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

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
    // how do I know????? (I cheated by looking the InfiniTensor Official Repo)
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
        CHECK_LE(sequence_length, 1024); // GPT-2: max_seq_length = 1024
        std::vector<uint16_t> vec(num_sequences * sequence_length);
        fin.read(reinterpret_cast<char *>(vec.data()), data_size_in_bytes);
        for (size_t i = 0; i < vec.size(); ++i) { dst[i] = static_cast<int64_t>(vec[i]); }
    } else if (text_file.type == TinyShakespeareType::kUINT32) {
        CHECK_LE(sequence_length, 8192); // LLaMA-3: max_seq_length = 8192
        std::vector<int32_t> vec(num_sequences * sequence_length);
        fin.read(reinterpret_cast<char *>(vec.data()), data_size_in_bytes);
        for (size_t i = 0; i < vec.size(); ++i) { dst[i] = static_cast<int64_t>(vec[i]); }
    }
    return text_file;
}
} // namespace

// TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
//     // =================================== 作业 ===================================
//     // TODO：初始化数据集实例
//     // HINT: 调用ReadTinyShakespeareFile加载数据文件
//     // =================================== 作业 ===================================
//     text_file_ = ReadTinyShakespeareFile(filepath, sequence_length);
//     // x: num_samples_ 是 num_sequence
//     // 要减一，因为
//     // 如果你有 N 个 tokens，把它切成若干长度为 sequence_length 的序列。
//     // 那么：
//     // 第一个 batch 用 tokens [0..sequence_length-1] 作为输入，预测 [1..sequence_length]。
//     // 第二个 batch 用 tokens [1..sequence_length] 作为输入，预测 [2..sequence_length+1]。
//     // …以此类推。
//     // 👉 你会发现：最后一个 token 没法预测“下一个”，因为数据已经没有了。
//     // 所以：
//     // 如果原来有 num_sequences 个片段，
//     // 实际上能用来训练的只有 num_sequences - 1。
//     num_samples_ = text_file_.dims[0]; 
//     sequence_size_in_bytes_ = sequence_length * kTypeToSize.at(text_file_.type);
// }

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

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
