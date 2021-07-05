//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_REPLAY_BUFFER_H
#define HIPC21_REPLAY_BUFFER_H

#include <map>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include "segment_tree.h"

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
    typedef std::map<std::string, DataSpec> str_to_dataspec;

    explicit ReplayBuffer(int64_t capacity,
                          const str_to_dataspec &data_spec,
                          int64_t batch_size) : m_capacity(capacity), m_batch_size(batch_size) {
        for (auto &it : data_spec) {
            auto name = it.first;
            auto shape = it.second.m_shape;
            shape.insert(shape.begin(), capacity);
            m_storage[name] = torch::zeros(shape, torch::TensorOptions().dtype(it.second.m_dtype));
        }
    }

    // pure virtual function
    virtual std::shared_ptr<torch::Tensor> generate_idx() const = 0;

    std::unique_ptr<str_to_tensor> sample() {
        auto idx = generate_idx();
        return this->operator[](*idx);
    }

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
    virtual void add_batch(const str_to_tensor &data) {
        int64_t batch_size = data.begin()->second.sizes()[0];
        if (m_ptr + batch_size > capacity()) {
            std::cout << "Reaches the end of the replay buffer" << std::endl;
        }
        for (auto &it : data) {
            AT_ASSERT(batch_size == it.second.sizes()[0]);
            if (m_ptr + batch_size > capacity()) {
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
    int64_t m_size{};
    int64_t m_ptr{};
};


class UniformReplayBuffer : public ReplayBuffer {
public:
    explicit UniformReplayBuffer(int64_t capacity, const std::map<std::string, DataSpec> &data_spec,
                                 int64_t batch_size)
            : ReplayBuffer(capacity, data_spec, batch_size) {

    }

    std::shared_ptr<torch::Tensor> generate_idx() const override {
        auto idx = torch::randint(size(), {m_batch_size}, torch::TensorOptions().dtype(torch::kInt64));
        return std::make_unique<torch::Tensor>(idx);
    }
};

class PrioritizedReplayBuffer : public ReplayBuffer {
public:
    explicit PrioritizedReplayBuffer(int64_t capacity, const std::map<std::string, DataSpec> &data_spec,
                                     int64_t batch_size, float alpha, float beta)
            : ReplayBuffer(capacity, data_spec, batch_size),
              m_segment_tree(capacity),
              m_alpha(alpha),
              m_beta(beta),
              m_max_priority(1.0),
              m_min_priority(1.0) {

    }

    std::shared_ptr<torch::Tensor> generate_idx() const override {
        // generate index according to the priority
        auto scalar = torch::rand({m_batch_size}) * m_segment_tree.reduce();
        auto idx = m_segment_tree.get_prefix_sum_idx(scalar);
        return idx;
    }

    std::shared_ptr<torch::Tensor> get_weights(const torch::Tensor &idx) const {
        auto weights = m_segment_tree.operator[](idx);
        *weights = torch::pow(*weights / m_min_priority, -m_beta);
        return weights;
    }

    void update_priorities(const torch::Tensor &idx, const torch::Tensor &priorities) {
        auto new_priority = torch::abs(priorities + 1e-6);
        m_max_priority = std::max(m_max_priority, torch::max(new_priority).item().toFloat());
        m_min_priority = std::min(m_min_priority, torch::min(new_priority).item().toFloat());
        m_segment_tree.set(idx, torch::pow(new_priority, m_alpha));
    }

    void add_batch(const str_to_tensor &data) override {
        int64_t batch_size = data.begin()->second.sizes()[0];
        auto priority = torch::pow(torch::ones({batch_size}) * m_max_priority, m_alpha);
        if (m_ptr + batch_size > capacity()) {
            std::cout << "Reaches the end of the replay buffer" << std::endl;
        }
        for (auto &it : data) {
            AT_ASSERT(batch_size == it.second.sizes()[0]);
            if (m_ptr + batch_size > capacity()) {
                m_storage[it.first].index_put_({Slice(m_ptr, None)},
                                               it.second.index({Slice(None, capacity() - m_ptr)}));
                m_storage[it.first].index_put_({Slice(None, batch_size - (capacity() - m_ptr))},
                                               it.second.index({Slice(capacity() - m_ptr, None)}));
            } else {
                m_storage[it.first].index_put_({Slice(m_ptr, m_ptr + batch_size)}, it.second);
            }
        }

        if (m_ptr + batch_size > capacity()) {
            m_segment_tree.set(torch::arange(m_ptr, capacity()),
                               priority.index({Slice(None, capacity() - m_ptr)}));
            m_segment_tree.set(torch::arange(batch_size - (capacity() - m_ptr)),
                               priority.index({Slice(capacity() - m_ptr, None)}));
        } else {
            auto index = torch::arange(m_ptr, m_ptr + batch_size);
            m_segment_tree.set(index, priority);
        }

        m_ptr = (m_ptr + batch_size) % capacity();
        m_size = std::min(m_size + batch_size, capacity());
    }


private:
    SegmentTree m_segment_tree;
    float m_alpha;
    float m_beta;
    float m_max_priority;
    float m_min_priority;
};

#endif //HIPC21_REPLAY_BUFFER_H
