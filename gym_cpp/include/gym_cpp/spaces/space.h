//
// Created by chi on 10/19/21.
//

#ifndef GYM_CPP_SPACE_H
#define GYM_CPP_SPACE_H

#include <torch/torch.h>

#include <utility>

namespace gym::space {
    enum Type {
        Box_t, Discrete_t
    };


    class Space {
    public:
        [[nodiscard]] virtual torch::Tensor sample() const = 0;

        [[nodiscard]] virtual bool contains(const torch::Tensor &x) const = 0;

        virtual Type get_type() = 0;

        void seed(uint64_t seed) {
//            gen.set_current_seed(seed);
        }

    protected:
        // the gen is simply a placeholder. Not sure how to set the seed due to lack of documentation
        torch::Generator gen;
    };


    class Box : public Space {
    public:
        explicit Box(torch::Tensor low, torch::Tensor high);

        [[nodiscard]] torch::Tensor sample() const override;

        [[nodiscard]] bool contains(const torch::Tensor &x) const override;

        [[nodiscard]] torch::IntArrayRef get_shape() const;

        [[nodiscard]] const torch::Tensor &get_low() const;

        [[nodiscard]] const torch::Tensor &get_high() const;

        Type get_type() override {
            return Box_t;
        }

    private:
        const torch::Tensor low;
        const torch::Tensor high;

    };


    class Discrete : public Space {
    public:
        explicit Discrete(int64_t n);

        [[nodiscard]] torch::Tensor sample() const override;

        [[nodiscard]] bool contains(const torch::Tensor &x) const override;

        [[nodiscard]] int64_t get_n() const;

        Type get_type() override {
            return Discrete_t;
        }

    private:
        int64_t n;
    };

}


#endif //GYM_CPP_SPACE_H
