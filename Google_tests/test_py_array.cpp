//
// Created by chi on 10/19/21.
//

#include "gtest/gtest.h"
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <vector>

namespace py = pybind11;

TEST(py_array, py_array_to_tensor) {
    py::scoped_interpreter guard{};
    py::module_ sys = py::module_::import("sys");
    py::print(sys.attr("executable"));
    py::module_ np = py::module_::import("numpy");
    py::print(np.attr("__version__"));
    auto zeros = np.attr("random").attr("randn")(3, 2).cast<py::array_t<double>>();
    torch::ScalarType dtype;
    if (py::isinstance<py::array_t<float>>(zeros)) {
        dtype = torch::kFloat32;
    } else if (py::isinstance<py::array_t<double>>(zeros)) {
        dtype = torch::kFloat64;
    } else if (py::isinstance<py::array_t<uint8_t>>(zeros)) {
        dtype = torch::kUInt8;
    } else {
        throw std::runtime_error("Unknown dtype");
    }

    torch::Tensor data = torch::from_blob(zeros.mutable_data(), torch::ArrayRef(zeros.shape(), zeros.ndim()), torch::TensorOptions().dtype(dtype));
    std::cout << data << std::endl;

    // construct py_array from tensor

}


