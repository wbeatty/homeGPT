#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "structure.hpp"

class Loader {
private:
    int fd = -1;
    const char* file_name;
    void* file_ptr = nullptr;
    size_t file_size = 0;

    std::unordered_map<std::string, std::pair<float*, std::vector<size_t>>> tensors;

    Tensor get_tensor(const std::string& name);

    bool opened = false;
    bool parsed = false;
    bool model_created = false;

public:
    Loader(const char* fn) : file_name(fn) {};
    ~Loader();

    bool open_file();
    bool parse_file();
    std::unique_ptr<Model> model(int n_layers = 12);
};
