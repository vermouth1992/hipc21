//
// Created by chi on 7/17/21.
//

#include "replay_buffer/replay_buffer_base.h"

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

    str_to_tensor ReplayBuffer::sample() {
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
    str_to_tensor ReplayBuffer::operator[](const torch::Tensor &idx) {
        str_to_tensor output;
        for (auto &it: m_storage) {
            output[it.first] = it.second.index({idx});
        }
        return output;
    }


    // get all the data
    str_to_tensor ReplayBuffer::get() {
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

}

