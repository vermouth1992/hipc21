//
// Created by chi on 10/29/21.
//

#ifndef HIPC21_PRIORITIZED_REPLAY_BUFFER_OPTIMIZED_H
#define HIPC21_PRIORITIZED_REPLAY_BUFFER_OPTIMIZED_H

/*
 * Prioritized Replay Buffer with global synchronization.
 */

#include "replay_buffer_base.h"
#include "segment_tree.h"
#include "thread"

namespace rlu::replay_buffer {

    template<class SegmentTreeClass>
    class PrioritizedReplayBufferOpt final : public ReplayBuffer {
    public:
        explicit PrioritizedReplayBufferOpt(int64_t capacity, const str_to_dataspec &data_spec,
                                            int64_t batch_size,
                                            float alpha) : ReplayBuffer(capacity, data_spec, batch_size),
                                                           m_alpha(alpha),
                                                           m_max_priority(1.0),
                                                           m_min_priority(1.0) {
            m_segment_tree = std::make_shared<SegmentTreeClass>(capacity);
        }


        str_to_tensor sample() override {
            std::thread::id this_id = std::this_thread::get_id();
            thread_mutex.lock();
            if (!this->mutexes.contains(this_id)) {
                mutexes[this_id] = std::make_unique<std::mutex>();
            }
            thread_mutex.unlock();

            // only lock index generation
            mutexes.at(this_id)->lock();
            // generate index can run in parallel, but not with update priority
            auto idx = generate_idx();
            mutexes.at(this_id)->unlock();
            // update the priority to zero to avoid sampling by other threads
            // to need to reverse it because the priority will be updated anyway.
            for (auto p = mutexes.begin(); p != mutexes.end(); p++) {
                p->second->lock();
            }
            this->update_priorities(idx, torch::zeros({this->m_batch_size},
                                                      torch::TensorOptions().dtype(torch::kFloat32)));
            for (auto p = mutexes.rbegin(); p != mutexes.rend(); p++) {
                p->second->unlock();
            }


            auto data = this->operator[](idx);
            data["idx"] = idx;

            return data;
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

            for (auto p = mutexes.begin(); p != mutexes.end(); p++) {
                p->second->lock();
            }
            // no need to lock
            this->update_priorities(idx, priorities);
            for (auto p = mutexes.rbegin(); p != mutexes.rend(); p++) {
                p->second->unlock();
            }
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
            std::thread::id this_id = std::this_thread::get_id();
            thread_mutex.lock();
            if (!this->mutexes.contains(this_id)) {
                mutexes[this_id] = std::make_unique<std::mutex>();
            }
            thread_mutex.unlock();

            int64_t batch_size = data.begin()->second.sizes()[0];
            // update priority
            torch::Tensor idx;
            for (auto p = mutexes.begin(); p != mutexes.end(); p++) {
                p->second->lock();
            }
            if (m_ptr + batch_size > capacity()) {
                idx = torch::cat({torch::arange(m_ptr, capacity()), torch::arange(batch_size - (capacity() - m_ptr))},
                                 0);
            } else {
                idx = torch::arange(m_ptr, m_ptr + batch_size);
            }
            for (auto p = mutexes.rbegin(); p != mutexes.rend(); p++) {
                p->second->unlock();
            }

            ReplayBuffer::add_batch(data);

            for (auto p = mutexes.begin(); p != mutexes.end(); p++) {
                p->second->lock();
            }
            this->update_priorities(idx, priorities);
            for (auto p = mutexes.rbegin(); p != mutexes.rend(); p++) {
                p->second->unlock();
            }

        }

    private:
        std::shared_ptr<SegmentTreeClass> m_segment_tree;
        float m_alpha;
        float m_max_priority;
        float m_min_priority;
        std::map<std::thread::id, std::unique_ptr<std::mutex>> mutexes;
        std::mutex thread_mutex;
    };
}


#endif //HIPC21_PRIORITIZED_REPLAY_BUFFER_OPTIMIZED_H
