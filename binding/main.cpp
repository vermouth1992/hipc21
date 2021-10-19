//
// Created by Chi Zhang on 10/18/21.
//

#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <iostream>
#include <torch/torch.h>

namespace py = pybind11;

namespace gym {

    py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    py::module_ gym_module = py::module_::import("gym");

    static torch::ScalarType get_py_array_shape(const py::array &array) {
        torch::ScalarType dtype;
        if (py::isinstance<py::array_t<float>>(array)) {
            dtype = torch::kFloat32;
        } else if (py::isinstance<py::array_t<double>>(array)) {
            dtype = torch::kFloat64;
        } else if (py::isinstance<py::array_t<uint8_t>>(array)) {
            dtype = torch::kUInt8;
        } else {
            throw std::runtime_error("Unknown dtype");
        }
        return dtype;
    }

    template<typename T>
    class Env {
    public:
        explicit Env(const std::string &env_name) {
            env = gym_module.attr("make")(env_name);
        }

        [[nodiscard]] virtual T py_array_to(py::array &a) = 0;

        [[nodiscard]] virtual py::array to_py_array(T &t) = 0;

        [[nodiscard]] T reset() const {
            auto obs_py = env.attr("reset")().cast<py::array>();
            return this->py_array_to(obs_py);
        }

        [[nodiscard]] std::tuple<py::array, double, bool> step(const T &action) const {

        }

    private:
        py::object env;
    };

    class PyArrayEnv : public Env<py::array> {
        [[nodiscard]] py::array py_array_to(py::array &a) override {
            return a;
        }

        [[nodiscard]] py::array to_py_array(py::array &t) override {
            return t;
        }
    };

    class TorchEnv : public Env<torch::Tensor> {
        [[nodiscard]] torch::Tensor py_array_to(py::array &array) override {
            auto dtype = get_py_array_shape(array);
            torch::Tensor data = torch::from_blob(array.mutable_data(), torch::ArrayRef(array.shape(), array.ndim()),
                                                  torch::TensorOptions().dtype(dtype));
            return data;
        }

        [[nodiscard]] py::array to_py_array(torch::Tensor &t) override {
            return {};
        }
    };

}


int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    py::module_ sys = py::module_::import("sys");
    py::print(sys.attr("executable"));
    py::module_ np = py::module_::import("numpy");
    py::print(np.attr("__version__"));
    py::module_ gym = py::module_::import("gym");
    py::object env = gym.attr("make")("Pong-v4");
    env.attr("seed")(10);
    auto obs = env.attr("reset")().cast<py::array>();
    if (py::isinstance<py::array_t<float>>(obs)) {
        std::cout << "float32" << std::endl;
        auto o = obs.cast<py::array_t<float>>();
        auto r = o.mutable_unchecked<1>();
    } else if (py::isinstance<py::array_t<double>>(obs)) {
        std::cout << "float64" << std::endl;
        obs = obs.cast<py::array_t<double>>();
    } else if (py::isinstance<py::array_t<uint8_t>>(obs)) {
        std::cout << "uint8" << std::endl;
//        obs = obs.cast<py::array_t<double>>();
    }


    std::cout << obs.shape(0) << std::endl;

    py::print("Hello, World!"); // use the Python API
}