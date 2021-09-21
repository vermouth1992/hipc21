//
// Created by Chi Zhang on 9/21/21.
//

#include "replay_buffer/uniform_replay_buffer.h"

namespace rlu::replay_buffer {

    UniformReplayBuffer::UniformReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec,
                                             int64_t batch_size)
            : ReplayBuffer(capacity, data_spec, batch_size) {

    }

    torch::Tensor UniformReplayBuffer::generate_idx() const {
        auto idx = torch::randint(size(), {m_batch_size}, torch::TensorOptions().dtype(torch::kInt64));
        return idx;
    }
}