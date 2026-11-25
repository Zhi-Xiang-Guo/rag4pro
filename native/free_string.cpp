#include <cstdlib>
// Release C API strings with the allocator of the producing shared library.
extern "C" void RAGFlow_FreeString(void* value) {
    std::free(value);
}
