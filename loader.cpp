#include "loader.hpp"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

Loader::~Loader() {
    if (opened) munmap(file_ptr, file_size);
    if (fd != -1) close(fd);
}

bool Loader::open_file() {
    fd = open(file_name, O_RDONLY);

    if (fd == -1) {
        perror("open");
        std::cout << "[MODEL] ERROR: CAN'T OPEN FILE\n";
        return false;
    }

    struct stat st;
    fstat(fd, &st);
    file_size = st.st_size;

    if (file_size == 0) {
        std::cout << "[MODEL] ERROR: FILE EMPTY\n";
        return false;
    }

    // NULL starting address, size = size, read instead of write, private mapping, file descriptor, starting position in file
    file_ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (file_ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        std::cout << "[MODEL] ERROR: MAP FAILED\n";
        return false;
    }

    opened = true;
    return true;
}


bool Loader::parse_file() {
    if (!opened) {
        return false;
    }

    uint64_t header_size;
    memcpy(&header_size, file_ptr, sizeof(header_size));

    if (header_size < 8) {
        return false;
    }

    const char* data = static_cast<const char*>(file_ptr);

    const char* json_start = data + 8;

    json j = json::parse(json_start, json_start + header_size);

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        std::cout << it.key() << " : " << it.value() << "\n";

        // check if h.N.attn.bias, then skip. we will create our own causal mask later
        // also skip metadata
        std::string name = it.key();
        if (name.size() > 10) {
            if (name.substr(name.size() - 10) == ".attn.bias" || name == "__metadata__") {
                // skip 
                continue;
            }
        }
        std::vector<uint64_t> data_offsets = it.value().at("data_offsets");
        float* start_ptr = (float*)(data + 8 + header_size + data_offsets[0]);

        std::vector<size_t> shape = it.value().at("shape");

        tensors[name] = std::make_pair(start_ptr, shape);
    }

    parsed = true;

    return true;
}

Tensor Loader::get_tensor(const std::string& name) {
    // .at() throws on a typo instead of silently inserting an empty entry
    const auto& [data, shape] = tensors.at(name);
    return Tensor{data, shape};
}

std::unique_ptr<Model> Loader::model(int n_layers) {

    std::unique_ptr<Model> model = std::make_unique<Model>();

    if (!opened || !parsed) {
        return model;
    }

    // load embeddings
    model->wte = get_tensor("wte.weight");
    model->wpe = get_tensor("wpe.weight");

    // suffix of "h.{i}.<suffix>" paired with the Block field it fills
    static const std::pair<const char*, Tensor Block::*> block_tensors[] = {
        {"ln_1.weight",          &Block::ln_1_weight},
        {"ln_1.bias",            &Block::ln_1_bias},
        {"attn.c_attn.weight",   &Block::c_attn_weight},
        {"attn.c_attn.bias",     &Block::c_attn_bias},
        {"attn.c_proj.weight",   &Block::attn_proj_weight},
        {"attn.c_proj.bias",     &Block::attn_proj_bias},
        {"ln_2.weight",          &Block::ln_2_weight},
        {"ln_2.bias",            &Block::ln_2_bias},
        {"mlp.c_fc.weight",      &Block::mlp_fc_weight},
        {"mlp.c_fc.bias",        &Block::mlp_fc_bias},
        {"mlp.c_proj.weight",    &Block::mlp_proj_weight},
        {"mlp.c_proj.bias",      &Block::mlp_proj_bias},
    };

    model->blocks.resize(n_layers);

    for (int i = 0; i < n_layers; ++i) {
        Block& block = model->blocks[i];
        std::string prefix = "h." + std::to_string(i) + ".";

        for (const auto& [suffix, field] : block_tensors) {
            block.*field = get_tensor(prefix + suffix);
        }
    }

    // load final layernorm
    model->ln_f_weight = get_tensor("ln_f.weight");
    model->ln_f_bias = get_tensor("ln_f.bias");

    model_created = true;

    return model;
}