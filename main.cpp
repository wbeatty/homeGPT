#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "structure.hpp"
#include "loader.hpp"

constexpr const char* file_name = "./model.safetensors";


int main() {
    // open the weights file
    Loader ld(file_name);

    if (!ld.open_file()) { throw std::runtime_error("[MAIN] ERROR: UNABLE TO OPEN FILE"); }
    if (!ld.parse_file()) { throw std::runtime_error("[MAIN] ERROR: UNABLE TO PARSE FILE"); }

    std::unique_ptr<Model> model = ld.model();
    if (model->map == nullptr) { throw std::runtime_error("[MAIN] ERROR: MODEL NOT CREATED"); }
    
    // now can get token embeddings

    
    // madvise(ptr, size, MADV_WILLNEED);   before first forward pass 

    return 0;
}