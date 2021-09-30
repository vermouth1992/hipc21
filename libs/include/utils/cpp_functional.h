//
// Created by Chi Zhang on 9/30/21.
//

#ifndef HIPC21_CPP_FUNCTIONAL_H
#define HIPC21_CPP_FUNCTIONAL_H

#include <map>
#include <unordered_map>
#include <any>

namespace rlu::cpp_functional {
    template<typename K, typename V>
    V get_with_default_val(const std::map<K, V> &m, const K &key, const V &default_val) {
        if (m.contains(key)) {
            return m.at(key);
        } else {
            return default_val;
        }
    }

    template<typename K, typename V>
    V get_with_default_val(const std::unordered_map<K, V> &m, const K &key, const V &default_val) {
        if (m.contains(key)) {
            return m.at(key);
        } else {
            return default_val;
        }
    }

    template<typename K, typename V>
    V get_any_with_default_val(const std::map<K, std::any> &m, const K &key, const V &default_val) {
        if (m.contains(key)) {
            return std::any_cast<V>(m.at(key));
        } else {
            return default_val;
        }
    }

    template<typename K, typename V>
    V get_any_with_default_val(const std::unordered_map<K, std::any> &m, const K &key, const V &default_val) {
        if (m.contains(key)) {
            return std::any_cast<V>(m.at(key));
        } else {
            return default_val;
        }
    }

}


#endif //HIPC21_CPP_FUNCTIONAL_H
