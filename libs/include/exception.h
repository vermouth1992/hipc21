//
// Created by Chi Zhang on 9/8/21.
//

#ifndef HIPC21_EXCEPTIONS_H
#define HIPC21_EXCEPTIONS_H

#include <string>

class NotImplemented : public std::logic_error {
public:
    NotImplemented(const std::string &name) : std::logic_error("Function " + name + " not yet implemented") {};
};

#endif //HIPC21_EXCEPTIONS_H
