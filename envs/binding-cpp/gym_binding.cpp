#include "gym/gym.h"
#include <curl/curl.h>
#include "fmt/core.h"
#include "base64.h"
#include <cstdio>
#include <iostream>

using nlohmann::json;

namespace Gym {

    static bool verbose = false;

    static torch::Tensor decode_base64_to_tensor(const std::string &str) {
        std::vector<char> f;
        // b64decode
        macaron::Base64::Decode(str, f);
        torch::Tensor x = torch::pickle_load(f).toTensor();
        return x;
    }

    static std::string require(const json &v, const std::string &k) {
        if (!v.is_object() || !v.contains(k))
            throw std::runtime_error("cannot find required parameter '" + k + "'");
        return v[k].get<std::string>();
    }

    static std::shared_ptr<Space> space_from_json(const json &j) {
        std::shared_ptr<Space> r(new Space);
        json v = j["info"];
        std::string type = require(v, "name");
        if (type == "Discrete") {
            r->type = Space::DISCRETE;
            r->discreet_n = v["n"].get<int>(); // will throw runtime_error if cannot be converted to int

        } else if (type == "Box") {
            r->type = Space::BOX;
            json shape = v["shape"];
            json low = v["low"];
            json high = v["high"];
            if (!shape.is_array())
                throw std::runtime_error("cannot parse box space (1)");
            // construct shape
            for (auto &s : shape) {
                int e = s.get<int>();
                r->box_shape.push_back(e);
            }

            r->box_low = decode_base64_to_tensor(low.get<std::string>());
            r->box_high = decode_base64_to_tensor(high.get<std::string>());

//            std::cout << "Box low: " << r->box_low << std::endl << "Box high: " << r->box_high << std::endl;


        } else {
            throw std::runtime_error("unknown space type '" + type + "'");
        }

        return r;
    }


// curl

    static std::size_t curl_save_to_string(void *buffer, std::size_t size, std::size_t nmemb, void *userp) {
        auto *str = static_cast<std::string *>(userp);
        const std::size_t bytes = nmemb * size;
        str->append(static_cast<char *>(buffer), bytes);
        return bytes;
    }

    class ClientReal : public Client, public std::enable_shared_from_this<ClientReal> {
    public:
        std::string addr;
        int port{};

        std::shared_ptr<CURL> h;
        std::shared_ptr<curl_slist> headers;
        std::vector<char> curl_error_buf;

        ClientReal() {
            CURL *c = curl_easy_init();
            curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1);
            curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, 3000);
            curl_easy_setopt(c, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
            curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, true);
            curl_easy_setopt(c, CURLOPT_SSL_VERIFYPEER, 0);
            curl_easy_setopt(c, CURLOPT_SSL_VERIFYHOST, 0);
            curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, &curl_save_to_string);
            curl_error_buf.assign(CURL_ERROR_SIZE, 0);
            curl_easy_setopt(c, CURLOPT_ERRORBUFFER, curl_error_buf.data());
//            h.reset(c, std::ptr_fun(curl_easy_cleanup));
            h.reset(c, [](CURL *curl) { curl_easy_cleanup(curl); });
//            headers.reset(curl_slist_append(0, "Content-Type: application/json"), std::ptr_fun(curl_slist_free_all));
            headers.reset(curl_slist_append(0, "Content-Type: application/json"),
                          [](struct curl_slist *l) { curl_slist_free_all(l); });
        }

        json GET(const std::string &route) {
            std::string url = "http://" + addr + route;
            if (verbose) printf("GET %s\n", url.c_str());
            curl_easy_setopt(h.get(), CURLOPT_URL, url.c_str());
            curl_easy_setopt(h.get(), CURLOPT_PORT, port);
            std::string answer;
            curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &answer);
            curl_easy_setopt(h.get(), CURLOPT_POST, 0);
            curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, 0);

            CURLcode r;
            r = curl_easy_perform(h.get());
            if (r) throw std::runtime_error(curl_error_buf.data());

            json j;
            throw_server_error_or_response_code(answer, j);
            return j;
        }

        json POST(const std::string &route, const std::string &post_data) {
            std::string url = "http://" + addr + route;
            if (verbose) printf("POST %s\n%s\n", url.c_str(), post_data.c_str());
            curl_easy_setopt(h.get(), CURLOPT_URL, url.c_str());
            curl_easy_setopt(h.get(), CURLOPT_PORT, port);
            std::string answer;
            curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &answer);
            curl_easy_setopt(h.get(), CURLOPT_POST, 1);
            curl_easy_setopt(h.get(), CURLOPT_POSTFIELDS, post_data.c_str());
            curl_easy_setopt(h.get(), CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t) post_data.size());
            curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, headers.get());

            CURLcode r = curl_easy_perform(h.get());
            if (r) throw std::runtime_error(curl_error_buf.data());

            json j;
            throw_server_error_or_response_code(answer, j);
            return j;
        }

        void throw_server_error_or_response_code(const std::string &answer, json &j) {
            long response_code;
            CURLcode r = curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &response_code);
            if (r) throw std::runtime_error(curl_error_buf.data());
            if (verbose) printf("%i\n%s\n", (int) response_code, answer.c_str());

            std::string parse_error;
            j = json::parse(answer);
            if (!j.is_object()) {
                parse_error = "top level json is not an object";
                parse_error += "original json that caused error: " + answer;
            }

            if (response_code != 200 && j.is_object() && j.contains("message")) {
                throw std::runtime_error(j["message"].get<std::string>());
            } else if (response_code != 200) {
                throw std::runtime_error("bad HTTP response code, and also cannot parse server message: " + answer);
            } else {
                // 200, but maybe invalid json
                if (!parse_error.empty())
                    throw std::runtime_error(parse_error);
            }
        }

        std::shared_ptr<Environment> make(const std::string &env_id) override;
    };

    std::shared_ptr<Client> client_create(const std::string &addr, int port) {
        std::shared_ptr<ClientReal> client(new ClientReal);
        client->addr = addr;
        client->port = port;
        return client;
    }


// environment

    class EnvironmentReal : public Environment {
    public:
        std::string instance_id;
        std::shared_ptr<ClientReal> client;
        std::shared_ptr<Space> space_act;
        std::shared_ptr<Space> space_obs;

        std::shared_ptr<Space> action_space() override {
            if (!space_act)
                space_act = space_from_json(client->GET("/v1/envs/" + instance_id + "/action_space"));
            return space_act;
        }

        std::shared_ptr<Space> observation_space() override {
            if (!space_obs)
                space_obs = space_from_json(client->GET("/v1/envs/" + instance_id + "/observation_space"));
            return space_obs;
        }

        static void observation_parse(const json &v, torch::Tensor &save_here) {
            save_here = decode_base64_to_tensor(v.get<std::string>());
        }

        void reset(State *save_initial_state_here) override {
            json ans = client->POST("/v1/envs/" + instance_id + "/reset/", "");
            observation_parse(ans["observation"], save_initial_state_here->observation);
        }

        void step(const torch::Tensor &action, bool render, State *save_state_here) override {
            json act_json;
            std::shared_ptr<Space> aspace = action_space();
            if (aspace->type == Space::DISCRETE) {
                act_json["action"] = action.item<int64_t>();
            } else if (aspace->type == Space::BOX) {
                // should have a more general function. Left for now.
                json &array = act_json["action"];
                for (int c = 0; c < (int) action.sizes()[0]; ++c)
                    array[c] = action[c].item<float>();
            } else {
                assert(0);
            }
            act_json["render"] = render;
            json ans = client->POST("/v1/envs/" + instance_id + "/step/", act_json.dump());
            observation_parse(ans["observation"], save_state_here->observation);
            save_state_here->done = ans["done"].get<bool>();
            save_state_here->timeout = ans["timeout"].get<bool>();
            save_state_here->reward = ans["reward"].get<float>();
        }
    };

    std::shared_ptr<Environment> ClientReal::make(const std::string &env_id) {
        json req;
        req["env_id"] = env_id;
        json ans = POST("/v1/envs/", req.dump());
        std::string instance_id = require(ans, "instance_id");
        if (verbose) printf(" * created %s instance_id=%s\n", env_id.c_str(), instance_id.c_str());
        std::shared_ptr<EnvironmentReal> env(new EnvironmentReal);
        env->client = shared_from_this();
        env->instance_id = instance_id;
        return env;
    }

} // namespace
