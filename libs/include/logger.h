//
// Created by chi on 7/23/21.
//

#ifndef HIPC21_LOGGER_H
#define HIPC21_LOGGER_H

#define FMT_HEADER_ONLY

#include <string>
#include <fstream>
#include <filesystem>
#include <map>
#include <utility>
#include <vector>
#include <numeric>
#include <optional>
#include <pthread.h>
#include "nlohmann/json.hpp"
#include "common.h"
#include "fmt/format.h"
#include "fmt/compile.h"


namespace rlu::logger {
    namespace fs = std::filesystem;
    using json = nlohmann::json;


    std::string setup_logger_kwargs(const std::string &exp_name, int64_t seed, const std::string &data_dir);

    class Logger {
    public:
        Logger() = default;

        ~Logger();

        explicit Logger(const std::string &output_dir = {}, std::string exp_name = {},
                        const std::string &output_fname = "progress.txt");

        static void log(const std::string &msg, const std::string &color = "green");

        void save_config(json &root);

        void log_tabular(const std::string &key, float val);

        virtual void dump_tabular();

    protected:
        const std::string m_output_dir;
        std::ofstream m_output_file;
        bool m_first_row = true;
        std::vector<std::string> m_log_headers;
        std::map<std::string, float> m_log_current_row;
        const std::string m_exp_name;
        const std::string delimiter = "\t";
        int64_t max_key_len{};
        int64_t n_slashes{};

    private:
        // use global locks to ensure thread safety
        pthread_mutex_t mutex{};

        void set_logger_style();
    };

    class EpochLogger final : public Logger {
        using Logger::Logger;
    public:
        void store(const std::string &name, const std::vector<float> &data);

        void store(const std::string &name, float data);

        void store(const std::map<std::string, std::vector<float>> &data);

        void log_tabular(const std::string &key, std::optional<float> val, bool with_min_and_max = false,
                         bool average_only = false);

        std::vector<float> get(const std::string &key);

        std::map<std::string, float> get_stats(const std::string &key, bool with_min_and_max, bool average_only);

        void dump_tabular() override;

    protected:
        std::map<std::string, std::vector<float>> m_epoch_dict;
    private:
        pthread_mutex_t mutex{};
    };

}

#endif //HIPC21_LOGGER_H
