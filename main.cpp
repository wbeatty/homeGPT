#include <sys/mman.h>
#include <iostream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <cstring>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// tensors[name] = (offset ptr, [shape])
std::unordered_map<std::string, std::pair<float*, std::vector<size_t>>> tensors;

void parse_file(void* ptr) {

    uint64_t header_size;
    memcpy(&header_size, ptr, sizeof(header_size));

    if (header_size < 8) {
        return;
    }

    const char* data = static_cast<const char*>(ptr);

    const char* json_start = data + 8;

    json j = json::parse(json_start, json_start + header_size);

    std::string ignore = ".attn.bias";

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        // check if h.N.attn.bias, then skip. we will create our own causal mask later
        std::string name = it.key();
        if (name.size() > 10) {
            std::string check_ignore = name.substr(name.size() - 10);
            if (ignore == check_ignore || name == "__metadata__") {
                // skip 
                continue;
            }
        }
        std::vector<uint64_t> data_offsets = it.value().at("data_offsets");
        float* start_ptr = (float*)(data + 8 + header_size + data_offsets[1]);

        std::vector<size_t> shape = it.value().at("shape");

        tensors[name] = std::make_pair(start_ptr, shape);
    }
}

int main() {
    // open the weights file
    int fd = open("./model.safetensors", O_RDONLY);

    if (fd == -1) {
        perror("open");
        return 1;
    }

    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;

    if (size == 0) {
        close(fd);
        return 0;
    }

    // NULL starting address, size = size, read instead of write, private mapping, file descriptor, starting position in file
    void* ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    parse_file(ptr);

    munmap(ptr, size);
    close(fd);

    return 0;
}