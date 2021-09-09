//
// Created by Chi Zhang on 9/8/21.
//

#ifndef HIPC21_EXCEPTIONS_H
#define HIPC21_EXCEPTIONS_H


class NotImplemented : public std::logic_error {
public:
    NotImplemented() : std::logic_error("Function not yet implemented") {};
};

#endif //HIPC21_EXCEPTIONS_H
