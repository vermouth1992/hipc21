//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_REPLAY_BUFFER_BASE_H
#define HIPC21_REPLAY_BUFFER_BASE_H

#include <map>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include "fmt/format.h"
#include "segment_tree_base.h"
#include "type.h"

using namespace torch::indexing;

namespace rlu::replay_buffer {

    class ReplayBuffer {
    public:
        explicit ReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec, int64_t batch_size);

        // pure virtual function
        [[nodiscard]] virtual torch::Tensor generate_idx() const = 0;

        virtual str_to_tensor sample();

        void reset();

        [[nodiscard]] int64_t size() const;

        [[nodiscard]] int64_t capacity() const;

        str_to_tensor operator[](const torch::Tensor &idx);

        str_to_tensor get();

        virtual void add_batch(const str_to_tensor &data);

        void add_single(const str_to_tensor &data);

    protected:
        str_to_tensor m_storage;
        int64_t m_capacity;
        int64_t m_batch_size;
        int64_t m_size{};
        int64_t m_ptr{};
    };
}

#endif //HIPC21_REPLAY_BUFFER_BASE_H
