//
// Created by chi on 7/5/21.
//

#ifndef HIPC21_COMMON_H
#define HIPC21_COMMON_H

#include <cstring>
#include <iostream>

#ifdef __APPLE__
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#elif __MINGW32__
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define MSG(x) std::cout << __FILENAME__ << ':' << __LINE__ << ':' << __func__ << "() : " << x << std::endl

static void M_Assert_(const char *expr_str, bool expr, const char *file, int line, const char *msg) {
    if (!expr) {
        std::cerr << "Assert failed:\t" << msg << "\n"
                  << "Expected:\t" << expr_str << "\n"
                  << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}


#define M_Assert(Expr, Msg) M_Assert_(#Expr, Expr, __FILE__, __LINE__, Msg)


#endif //HIPC21_COMMON_H
