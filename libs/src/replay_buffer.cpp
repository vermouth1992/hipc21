//
// Created by chi on 7/17/21.
//

#include "replay_buffer/replay_buffer.h"

namespace rlu::replay_buffer {
    ReplayBuffer::ReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec,
                               int64_t batch_size) :
            m_capacity(capacity),
            m_batch_size(batch_size) {
        for (auto &it: data_spec) {
            auto name = it.first;
            auto shape = it.second.m_shape;
            shape.insert(shape.begin(), capacity);
            m_storage[name] = torch::zeros(shape, torch::TensorOptions().dtype(it.second.m_dtype));
        }
    }

    ReplayBuffer::str_to_tensor ReplayBuffer::sample() {
        auto idx = generate_idx();
        return this->operator[](idx);
    }

    void ReplayBuffer::reset() {
        m_size = 0;
        m_ptr = 0;
    }

    int64_t ReplayBuffer::size() const {
        return m_size;
    }

    int64_t ReplayBuffer::capacity() const {
        return m_capacity;
    }

    // get data by index
    ReplayBuffer::str_to_tensor ReplayBuffer::operator[](const torch::Tensor &idx) {
        str_to_tensor output;
        for (auto &it: m_storage) {
            output[it.first] = it.second.index({idx});
        }
        return output;
    }


    // get all the data
    ReplayBuffer::str_to_tensor ReplayBuffer::get() {
        torch::Tensor idx = torch::arange(m_size);
        return this->operator[](idx);
    }

    // add data samples
    void ReplayBuffer::add_batch(const str_to_tensor &data) {
        int64_t batch_size = data.begin()->second.sizes()[0];
        if (m_ptr + batch_size > capacity()) {
            std::cout << "Reaches the end of the replay buffer" << std::endl;
        }
        for (auto &it: data) {
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
    void ReplayBuffer::add_single(const str_to_tensor &data) {
        // add batch dimension
        str_to_tensor batch_data;
        for (auto &it: data) {
            batch_data[it.first] = it.second.unsqueeze(0);
        }
        this->add_batch(batch_data);
    }

    UniformReplayBuffer::UniformReplayBuffer(int64_t capacity, const std::map<std::string, DataSpec> &data_spec,
                                             int64_t batch_size)
            : ReplayBuffer(capacity, data_spec, batch_size) {

    }

    torch::Tensor UniformReplayBuffer::generate_idx() const {
        auto idx = torch::randint(size(), {m_batch_size}, torch::TensorOptions().dtype(torch::kInt64));
        return idx;
    }

    PrioritizedReplayBuffer::PrioritizedReplayBuffer(int64_t capacity, const std::map<std::string, DataSpec> &data_spec,
                                                     int64_t batch_size, float alpha, const std::string &segment_tree)
            : ReplayBuffer(capacity, data_spec, batch_size),
              m_alpha(alpha),
              m_max_priority(1.0),
              m_min_priority(1.0) {
        if (segment_tree == "cpp") {
            m_segment_tree = std::make_shared<SegmentTreeCPP>(capacity);
        } else if (segment_tree == "torch") {
            m_segment_tree = std::make_shared<SegmentTreeTorch>(capacity);
        } else if (segment_tree == "nary") {
            m_segment_tree = std::make_shared<SegmentTreeNary>(capacity);
        } else if (segment_tree == "fpga") {
            m_segment_tree = std::make_shared<SegmentTreeFPGA>(capacity);
        } else {
            throw std::runtime_error(fmt::format("Unknown segment tree implementation {}", segment_tree));
        }

    }

    torch::Tensor PrioritizedReplayBuffer::generate_idx() const {
        // generate index according to the priority
        return m_segment_tree->sample_idx(m_batch_size);
        auto scalar = torch::rand({m_batch_size}) * m_segment_tree->reduce();
        auto idx = m_segment_tree->get_prefix_sum_idx(scalar);
        return idx;
    }

    torch::Tensor PrioritizedReplayBuffer::get_weights(const torch::Tensor &idx, const float beta) const {
        auto weights = m_segment_tree->operator[](idx);
        weights = torch::pow(weights * ((float) size() / m_segment_tree->reduce()), -beta);
        weights = weights / torch::max(weights);
        return weights;
    }

    void PrioritizedReplayBuffer::update_priorities(const torch::Tensor &idx, const torch::Tensor &priorities) {
        auto new_priority = torch::pow(torch::abs(priorities + 1e-6), m_alpha);
        m_max_priority = std::max(m_max_priority, torch::max(new_priority).item().toFloat());
        m_min_priority = std::min(m_min_priority, torch::min(new_priority).item().toFloat());
        m_segment_tree->set(idx, new_priority);
    }

    // default priority to the maximum priority
    void PrioritizedReplayBuffer::add_batch(const str_to_tensor &data) {
        int64_t batch_size = data.begin()->second.sizes()[0];
        auto priorities = torch::ones({batch_size}) * m_max_priority;
        this->add_batch(data, priorities);
    }

    void PrioritizedReplayBuffer::add_batch(const ReplayBuffer::str_to_tensor &data, const torch::Tensor &priorities) {
        int64_t batch_size = data.begin()->second.sizes()[0];
        if (m_ptr + batch_size > capacity()) {
            std::cout << "Reaches the end of the replay buffer" << std::endl;
        }
        for (auto &it: data) {
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

        // update priority
        torch::Tensor idx;
        if (m_ptr + batch_size > capacity()) {
            idx = torch::cat({torch::arange(m_ptr, capacity()), torch::arange(batch_size - (capacity() - m_ptr))}, 0);
        } else {
            idx = torch::arange(m_ptr, m_ptr + batch_size);
        }

        this->update_priorities(idx, priorities);

        m_ptr = (m_ptr + batch_size) % capacity();
        m_size = std::min(m_size + batch_size, capacity());
    }

}

