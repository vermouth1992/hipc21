//
// Created by chi on 7/23/21.
//

#ifndef HIPC21_LOGGER_H
#define HIPC21_LOGGER_H

#include <string>
#include <fstream>
#include <filesystem>
#include <map>
#include <vector>

namespace fs = std::filesystem;


std::string colorize(const std::string &string, const std::string &color, bool bold, bool highlight) {
    return std::string();
}

class Logger {
public:
    Logger() = default;

    ~Logger() {
        m_output_file.close();
    }

    explicit Logger(const std::string &output_dir = {}, const std::string &output_fname = "progress.txt",
                    const std::string &exp_name = {}) :
            m_output_dir(output_dir) {
        if (!output_dir.empty()) {
            if (fs::exists(output_dir)) {
                std::cout << "Warning: Log dir " << output_dir << " already exists! Storing info there anyway."
                          << std::endl;
            } else {
                fs::create_directories(output_dir);
            }
            const std::string file_name = "output_dir + \"/\" + output_fname";
            m_output_file.open(file_name, std::fstream::out);
            std::cout << "Logging data to " << file_name << std::endl;
        }

    }

protected:
    const std::string m_output_dir;
    std::ofstream m_output_file;
    bool m_first_row;
    std::vector<std::string> m_log_headers;

};

class EpochLogger : public Logger {
    using Logger::Logger;
};


#endif //HIPC21_LOGGER_H
