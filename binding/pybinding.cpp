//
// Created by chi on 7/17/21.
//

#include "replay_buffer/SumTree.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

template<typename T>
void declare_array(py::module &m, const std::string &typestr) {
    using Class = SumTree<T>;
    std::string pyclass_name = std::string("SumTree") + typestr;
    py::class_<Class>(m, pyclass_name.c_str())
            .def(py::init<int64_t, int64_t>())
//            .def("size", &Class::size)
//            .def("set", &Class::set)
            ;
}

// Python binding
PYBIND11_MODULE(sumtree, m) {
    declare_array<float>(m, "float");
}