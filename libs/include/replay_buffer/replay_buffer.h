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

namespace rlu::replay_buffer {

    struct DataSpec {
        torch::Dtype m_dtype;
        std::vector<int64_t> m_shape;

        DataSpec(std::vector<int64_t> shape, torch::Dtype dtype) : m_dtype(dtype), m_shape(std::move(shape)) {

        }
    };

    class ReplayBuffer {
    public:
        typedef std::map<std::string, torch::Tensor> str_to_tensor;
        typedef std::map<std::string, DataSpec> str_to_dataspec;

        explicit ReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec, int64_t batch_size);

        // pure virtual function
        [[nodiscard]] virtual std::shared_ptr<torch::Tensor> generate_idx() const = 0;

        std::unique_ptr<str_to_tensor> sample();

        void reset();

        [[nodiscard]] int64_t size() const;

        [[nodiscard]] int64_t capacity() const;

        std::unique_ptr<str_to_tensor> operator[](const torch::Tensor &idx);

        std::unique_ptr<str_to_tensor> get();

        virtual void add_batch(const str_to_tensor &data);

        void add_single(const str_to_tensor &data);

    protected:
        str_to_tensor m_storage;
        int64_t m_capacity;
        int64_t m_batch_size;
        int64_t m_size{};
        int64_t m_ptr{};
    };


    class UniformReplayBuffer final : public ReplayBuffer {
    public:
        explicit UniformReplayBuffer(int64_t capacity, const std::map<std::string, DataSpec> &data_spec,
                                     int64_t batch_size);

        [[nodiscard]] std::shared_ptr<torch::Tensor> generate_idx() const override;


    };


    class PrioritizedReplayBuffer final : public ReplayBuffer {
    public:
        explicit PrioritizedReplayBuffer(int64_t capacity, const std::map<std::string, DataSpec> &data_spec,
                                         int64_t batch_size, float alpha);

        [[nodiscard]] std::shared_ptr<torch::Tensor> generate_idx() const override;

        [[nodiscard]] std::shared_ptr<torch::Tensor> get_weights(const torch::Tensor &idx, float beta) const;

        void update_priorities(const torch::Tensor &idx, const torch::Tensor &priorities);

        void add_batch(const str_to_tensor &data) override;

    private:
        SegmentTreeCPP m_segment_tree;
        float m_alpha;
        float m_max_priority;
        float m_min_priority;
    };
}

#endif //HIPC21_REPLAY_BUFFER_H