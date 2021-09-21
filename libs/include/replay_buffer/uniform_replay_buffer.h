//
// Created by Chi Zhang on 9/21/21.
//

#ifndef HIPC21_UNIFORM_REPLAY_BUFFER_H
#define HIPC21_UNIFORM_REPLAY_BUFFER_H

#include "replay_buffer_base.h"

namespace rlu::replay_buffer {
    class UniformReplayBuffer final : public ReplayBuffer {
    public:
        explicit UniformReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec, int64_t batch_size);

        [[nodiscard]] torch::Tensor generate_idx() const override;


    };
}


#endif //HIPC21_UNIFORM_REPLAY_BUFFER_H
