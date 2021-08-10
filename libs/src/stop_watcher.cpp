//
// Created by Chi Zhang on 8/5/21.
//

#include "utils/stop_watcher.h"

namespace rlu::watcher {
    StopWatcher::StopWatcher(std::string name) : m_elapsed(0), m_name(std::move(name)) {

    }

    StopWatcher::StopWatcher() : m_elapsed(0), m_name("default") {

    }

    std::string StopWatcher::name() const {
        return m_name;
    }

    void StopWatcher::reset() {
        m_elapsed = 0;
    }

    void StopWatcher::start() {
        m_start_time = std::chrono::high_resolution_clock::now();
    }

    void StopWatcher::lap() {
        // not accumulative
        auto end_time = std::chrono::high_resolution_clock::now();
        m_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - m_start_time).count();
    }

    void StopWatcher::stop() {
        // accumulate. Need to restart
        auto end_time = std::chrono::high_resolution_clock::now();
        m_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - m_start_time).count();
    }

    int64_t StopWatcher::nanoseconds() const {
        return m_elapsed;
    }

    double StopWatcher::microseconds() const {
        return (double) nanoseconds() / 1000.;
    }

    double StopWatcher::milliseconds() const {
        return (double) nanoseconds() / 1000000.;
    }

    double StopWatcher::seconds() const {
        return (double) nanoseconds() / 1000000000.;
    }


}

