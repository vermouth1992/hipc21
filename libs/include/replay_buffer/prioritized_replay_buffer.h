//
// Created by Chi Zhang on 9/21/21.
//

#ifndef HIPC21_PRIORITIZED_REPLAY_BUFFER_H
#define HIPC21_PRIORITIZED_REPLAY_BUFFER_H

#include "replay_buffer_base.h"
#include "segment_tree.h"

namespace rlu::replay_buffer {

    template<class SegmentTreeClass>
    class PrioritizedReplayBuffer final : public ReplayBuffer {
    public:
        explicit PrioritizedReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec, int64_t batch_size,
                                         float alpha) : ReplayBuffer(capacity, data_spec, batch_size),
                                                        m_alpha(alpha),
                                                        m_max_priority(1.0),
                                                        m_min_priority(1.0) {
            m_segment_tree = std::make_shared<SegmentTreeClass>(capacity);
        }

        str_to_tensor operator[](const torch::Tensor &idx) const override {
            auto result = ReplayBuffer::operator[](idx);
            result["weights"] = this->get_weights(idx, 1.0);
            return result;
        }

        [[nodiscard]] torch::Tensor generate_idx() const override {
            // generate index according to the priority
            return m_segment_tree->sample_idx(m_batch_size);
        }

        [[nodiscard]] torch::Tensor get_weights(const torch::Tensor &idx, float beta) const {
            auto weights = m_segment_tree->operator[](idx);
            weights = torch::pow(weights * ((float) size() / m_segment_tree->reduce()), -beta);
            weights = weights / torch::max(weights);
            return weights;
        }

        void update_priorities(const torch::Tensor &idx, const torch::Tensor &priorities) {
            m_max_priority = std::max(m_max_priority, torch::max(priorities).item().toFloat());
            m_min_priority = std::min(m_min_priority, torch::min(priorities).item().toFloat());
            auto new_priority = torch::pow(torch::abs(priorities + 1e-2), m_alpha);
            m_segment_tree->set(idx, new_priority);
        };

        void post_process(str_to_tensor &data) override {
            auto idx = data.at("idx");
            auto priorities = data.at("priority");
            this->update_priorities(idx, priorities);
        }

        void add_batch(str_to_tensor &data) override {
            int64_t batch_size = data.begin()->second.sizes()[0];
            torch::Tensor priorities;
            if (data.contains("priority")) {
                priorities = data.at("priority");
                data.erase("priority");
            } else {
                priorities = torch::ones({batch_size}) * m_max_priority;
            }

            this->add_batch(data, priorities);
        };

        void add_batch(str_to_tensor &data, torch::Tensor &priorities) {
            int64_t batch_size = data.begin()->second.sizes()[0];
            // update priority
            torch::Tensor idx;
            if (m_ptr + batch_size > capacity()) {
                idx = torch::cat({torch::arange(m_ptr, capacity()), torch::arange(batch_size - (capacity() - m_ptr))},
                                 0);
            } else {
                idx = torch::arange(m_ptr, m_ptr + batch_size);
            }
            this->update_priorities(idx, priorities);
            ReplayBuffer::add_batch(data);
        }

    private:
        std::shared_ptr<SegmentTreeClass> m_segment_tree;
        float m_alpha;
        float m_max_priority;
        float m_min_priority;
    };
}


#endif //HIPC21_PRIORITIZED_REPLAY_BUFFER_H
