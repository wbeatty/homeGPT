#pragma once

#include <cstddef>
#include <sys/mman.h>
#include <vector>

struct Tensor {
    float* data;
    std::vector<size_t> shape;
};

struct Block {
    Tensor ln_1_weight, ln_1_bias;
    Tensor c_attn_weight, c_attn_bias;
    Tensor attn_proj_weight, attn_proj_bias;
    Tensor ln_2_weight, ln_2_bias;
    Tensor mlp_fc_weight, mlp_fc_bias;
    Tensor mlp_proj_weight, mlp_proj_bias;
};

struct Model {
    void* map = nullptr;
    size_t size;

    Tensor wte, wpe;
    std::vector<Block> blocks;
    Tensor ln_f_weight, ln_f_bias;

    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model() = default;
};