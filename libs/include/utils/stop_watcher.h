//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_STOP_WATCHER_H
#define HIPC21_STOP_WATCHER_H


#include <string>

namespace rlu::watcher {
    class StopWatcher {
    public:
        explicit StopWatcher(std::string name);

        explicit StopWatcher();

        [[nodiscard]] std::string name() const;

        void reset();

        void start();

        void lap();

        void stop();

        [[nodiscard]] int64_t nanoseconds() const;

        [[nodiscard]] double microseconds() const;

        [[nodiscard]] double milliseconds() const;

        [[nodiscard]] double seconds() const;


    private:
        int64_t m_elapsed;
        std::string m_name;
        std::chrono::high_resolution_clock::time_point m_start_time;
    };

}


#endif //HIPC21_STOP_WATCHER_H
