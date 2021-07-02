//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_REPLAY_BUFFER_H
#define HIPC21_REPLAY_BUFFER_H

#include <map>
#include <torch/torch.h>
#include <utility>
#include <vector>

using namespace torch::indexing;

struct DataSpec {
    torch::Dtype m_dtype;
    std::vector<int64_t> m_shape;

    DataSpec(std::vector<int64_t> shape, torch::Dtype dtype) : m_shape(std::move(shape)), m_dtype(dtype) {

    }
};

class ReplayBuffer {
public:
    typedef std::map<std::string, torch::Tensor> str_to_tensor;

    explicit ReplayBuffer(uint capacity,
                          const std::map<std::string, DataSpec> &data_spec,
                          uint batch_size) : m_capacity(capacity), m_batch_size(batch_size) {
        for (auto &it : data_spec) {
            auto name = it.first;
            auto shape = it.second.m_shape;
            shape.insert(shape.begin(), capacity);
            m_storage[name] = torch::zeros(shape, torch::TensorOptions().dtype(it.second.m_dtype));
        }
    }

    // pure virtual function
    virtual std::unique_ptr<str_to_tensor> sample() = 0;

    void reset() {
        m_size = 0;
        m_ptr = 0;
    }

    int64_t size() const {
        return m_size;
    }

    int64_t capacity() const {
        return m_capacity;
    }

    // get data by index
    std::unique_ptr<str_to_tensor> operator[](const torch::Tensor &idx) {
        str_to_tensor output;
        for (auto &it : m_storage) {
            output[it.first] = it.second.index({idx});
        }
        return std::make_unique<str_to_tensor>(output);
    }

    // get all the data
    std::unique_ptr<str_to_tensor> get() {
        torch::Tensor idx = torch::arange(m_size);
        return this->operator[](idx);
    }

    // add data samples
    void add_batch(const str_to_tensor &data) {
        int64_t batch_size = data.begin()->second.sizes()[0];
        for (auto &it : data) {
            AT_ASSERT(batch_size == it.second.sizes()[0]);
            if (m_ptr + batch_size > capacity()) {
                std::cout << "Reaches the end of the replay buffer" << std::endl;
                m_storage[it.first].index_put_({Slice(m_ptr, None)},
                                               it.second.index({Slice(None, capacity() - m_ptr)}));
                m_storage[it.first].index_put_({Slice(None, batch_size - (capacity() - m_ptr))},
                                               it.second.index({Slice(capacity() - m_ptr, None)}));
            } else {
                m_storage[it.first].index_put_({Slice(m_ptr, m_ptr + batch_size)}, it.second);
            }
        }
        m_ptr = (m_ptr + batch_size) % capacity();
        m_size = std::min(m_size + batch_size, capacity());
    }

    // add one data sample
    void add_single(const str_to_tensor &data) {
        // add batch dimension
        str_to_tensor batch_data;
        for (auto &it: data) {
            batch_data[it.first] = it.second.unsqueeze_(0);
        }
        this->add_batch(batch_data);
    }

protected:
    str_to_tensor m_storage;
    int64_t m_capacity;
    int64_t m_batch_size;
    int64_t m_size;
    int64_t m_ptr;
};


class UniformReplayBuffer : public ReplayBuffer {
    std::unique_ptr<str_to_tensor> sample() override {
        auto idx = torch::randint(size(), {m_batch_size});
        return this->operator[](idx);
    }
};

class PrioritizedReplayBuffer : public ReplayBuffer {

};

#endif //HIPC21_REPLAY_BUFFER_H
