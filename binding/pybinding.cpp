//
// Created by chi on 7/17/21.
//

#include "SumTree.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

template<typename T>
void declare_array(py::module &m, const std::string &typestr) {
    using Class = SumTree<T>;
    std::string pyclass_name = std::string("SumTree") + typestr;
    py::class_<Class>(m, pyclass_name.c_str())
            .def(py::init<int64_t, int64_t>())
            .def("size", &Class::size)
            .def("set", &Class::set, py::arg("idx"), py::arg("value"))
            .def("__getitem__", py::vectorize(&Class::operator[]))
            .def("vector_set", py::vectorize(&Class::set))
            .def("get_prefix_sum_idx", &Class::get_prefix_sum_idx)
            .def("vector_get_prefix_sum_idx", py::vectorize(&Class::get_prefix_sum_idx))
            .def("reduce", py::overload_cast<>(&Class::reduce, py::const_));
}

// Python binding
PYBIND11_MODULE(sumtree, m) {
    declare_array<float>(m, "float");
}