//
// Created by chi on 7/5/21.
//

#ifndef HIPC21_COMMON_H
#define HIPC21_COMMON_H

#include <cstring>

#ifdef __APPLE__
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#elif __MINGW32__
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define MSG(x) std::cout << __FILENAME__ << ':' << __LINE__ << ':' << __func__ << "() : " << x << std::endl

#endif //HIPC21_COMMON_H
