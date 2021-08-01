//
// Created by chi on 7/23/21.
//

#ifndef HIPC21_LOGGER_H
#define HIPC21_LOGGER_H

#include <string>
#include <fstream>
#include <filesystem>
#include <map>
#include <utility>
#include <vector>
#include <numeric>
#include <optional>
#include "nlohmann/json.hpp"
#include "common.h"
#include "fmt/core.h"
#include "fmt/compile.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::map<std::string, int> color2num = {{"gray",    30},
                                               {"red",     31},
                                               {"green",   32},
                                               {"yellow",  33},
                                               {"blue",    34},
                                               {"magenta", 35},
                                               {"cyan",    36},
                                               {"white",   37},
                                               {"crimson", 38}};

static std::string string_join(const std::vector<std::string> &x, const std::string &delimiter) {
    return std::accumulate(std::begin(x), std::end(x), std::string(),
                           [delimiter](const std::string &ss, const std::string &s) {
                               return ss.empty() ? s : ss + delimiter + s;
                           });
}

static std::string colorize(const std::string &string, const std::string &color,
                            bool bold = false, bool highlight = false) {
    std::vector<std::string> attr;
    int num = color2num.at(color);
    if (highlight) num += 10;
    attr.push_back(std::to_string(num));
    if (bold) attr.emplace_back("1");
    return fmt::format("\x1b[{}m{}\x1b[0m", string_join(attr, ";"), string);
}


template<class T>
static std::map<std::string, float> statistics_scalar(const std::vector<T> &v,
                                                      bool average_only = false,
                                                      bool with_min_and_max = false) {
    float sum = std::accumulate(v.begin(), v.end(), 0.0);
    float mean = sum / v.size();
    std::map<std::string, float> result{{"Average", mean}};

    if (!average_only) {
        float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
        float stddev = std::sqrt(sq_sum / v.size() - mean * mean);
        result["Std"] = stddev;
    }

    if (with_min_and_max) {
        result["Max"] = *std::max_element(v.begin(), v.end());
        result["Min"] = *std::min_element(v.begin(), v.end());
    }
    return result;
}

static std::string setup_logger_kwargs(
        const std::string &exp_name,
        int seed,
        const std::string &data_dir
) {
    /*
     Sets up the output_dir for a logger and returns a dict for logger kwargs.

    output_dir = data_dir/exp_name/exp_name_s[seed]

    Args:

        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
     */
    auto output_dir = fs::path(data_dir);
    output_dir /= exp_name;
    output_dir /= exp_name + "_s" + fmt::to_string(seed);
    return output_dir;
}

class Logger {
public:
    Logger() = default;

    ~Logger() {
        m_output_file.close();
    }

    explicit Logger(const std::string &output_dir = {}, std::string exp_name = {},
                    const std::string &output_fname = "progress.txt") :
            m_output_dir(output_dir),
            m_first_row(true),
            m_exp_name(std::move(exp_name)) {
        if (!output_dir.empty()) {
            if (fs::exists(output_dir)) {
                std::cout << "Warning: Log dir " << output_dir << " already exists! Storing info there anyway."
                          << std::endl;
            } else {
                fs::create_directories(output_dir);
            }
            const std::string file_name = fs::path(output_dir) / output_fname;
            m_output_file.open(file_name, std::fstream::out);
            printf("%s", colorize("Logging data to " + file_name + "\n", "green").c_str());
        }
    }

    static void log(const std::string &msg, const std::string &color = "green") {
        fmt::print("{}", colorize(msg, color, true) + "\n");
    }

    void save_config(json &root) {
        if (!m_exp_name.empty()) {
            root["exp_name"] = m_exp_name;
        }
        std::string output = root.dump(4);
        fmt::print("{}", colorize("Saving config:\n", "cyan", true));
        fmt::print("{}\n", output);
        if (!m_output_dir.empty()) {
            std::ofstream f;
            f.open(fs::path(m_output_dir) / "config.json");
            f << output;
            f.close();
        }
    }


    void log_tabular(const std::string &key, float val) {
        if (m_first_row) {
            m_log_headers.push_back(key);
        } else {
            M_Assert(std::find(m_log_headers.begin(), m_log_headers.end(), key) != m_log_headers.end(),
                     fmt::format("Trying to introduce a new key {} that you didn't include in the first iteration\n",
                                 key).c_str());
        }
        M_Assert(m_log_current_row.find(key) == m_log_current_row.end(),
                 fmt::format("You already set {} this iteration. Maybe you forgot to call dump_tabular()\n",
                             key).c_str());
        m_log_current_row[key] = val;
    }

    virtual void dump_tabular() {
        const std::string delimiter = "\t";

        std::vector<int> key_lens;
        for (auto &key : m_log_headers) {
            key_lens.push_back((int) key.size());
        }
        auto max_elem = *std::max_element(key_lens.begin(), key_lens.end());
        auto max_key_len = std::max(15, max_elem);
        auto n_slashes = 22 + max_key_len;

        // header of the file
        if (!m_output_dir.empty()) {
            if (m_first_row) {
                auto s = string_join(m_log_headers, delimiter);
                m_output_file << s << std::endl;
            }
        }

        // print to console
        for (int i = 0; i < n_slashes; i++) {
            fmt::print("-");
        }
        fmt::print("\n");
        for (unsigned long i = 0; i < m_log_headers.size(); i++) {
            auto key = m_log_headers.at(i);
            auto val = m_log_current_row.at(key);
            fmt::print("| {:>{}} | {:>15} |\n", key, max_key_len, val);
            if (!m_output_dir.empty()) {
                m_output_file << val;
                if (i != m_log_headers.size() - 1) {
                    m_output_file << "\t";
                }
            }
        }
        if (!m_output_dir.empty()) {
            m_output_file << std::endl;
        }
        for (int i = 0; i < n_slashes; i++) {
            fmt::print("-");
        }
        fmt::print("\n");

        // write to file
        m_log_current_row.clear();
        m_first_row = false;
    }

protected:
    const std::string m_output_dir;
    std::ofstream m_output_file;
    bool m_first_row = true;
    std::vector<std::string> m_log_headers;
    std::map<std::string, float> m_log_current_row;
    const std::string m_exp_name;
};

class EpochLogger : public Logger {
    using Logger::Logger;
public:
    void store(const std::map<std::string, std::vector<float>> &data) {
        for (const auto &it : data) {
            if (!m_epoch_dict.contains(it.first)) {
                m_epoch_dict[it.first] = std::vector<float>();
            }
            m_epoch_dict[it.first].insert(m_epoch_dict[it.first].end(),
                                          it.second.begin(),
                                          it.second.end());
        }
    }

    void log_tabular(const std::string &key, std::optional<float> val, bool with_min_and_max, bool average_only) {
        if (val != std::nullopt) {
            Logger::log_tabular(key, val.value());
        } else {
            auto v = m_epoch_dict.at(key);
            auto stats = statistics_scalar(v, average_only, with_min_and_max);
            for (const auto &it : stats) {
                Logger::log_tabular(it.first + key, it.second);
            }
            m_epoch_dict[key].clear();
        }
    }

    std::vector<float> get(const std::string &key) const {
        return m_epoch_dict[key];
    }

    std::map<std::string, float> get_stats(const std::string &key, bool with_min_and_max, bool average_only) const {
        auto v = m_epoch_dict.at(key);
        auto stats = statistics_scalar(v, average_only, with_min_and_max);
        return stats;
    }

    void dump_tabular() override {
        Logger::dump_tabular();
        for (const auto &it : m_epoch_dict) {
            M_Assert(it.second.empty(), fmt::format("Key {} is not called using log_tabular\n", it.first).c_str());
        }
    }

protected:
    std::map<std::string, std::vector<float>> m_epoch_dict;
};


#endif //HIPC21_LOGGER_H
