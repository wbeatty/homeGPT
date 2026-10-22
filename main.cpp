#include <sys/mman.h>
#include <iostream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void parse_file(void* ptr) {

    uint64_t header_size;
    memcpy(&header_size, ptr, sizeof(header_size));

    if (header_size < 8) {
        return;
    }

    const char* data = static_cast<const char*>(ptr);

    const char* json_start = data + 8;

    json j = json::parse(json_start, json_start + header_size);

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        std::cout << it.key() << " : " << it.value() << "\n";
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