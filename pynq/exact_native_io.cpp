#include <cstddef>
#include <cstdint>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// One channel -> eight-lane device words. Quantized codes remain unchanged.
API int segmentation_pack_exact(const uint8_t* input, size_t pixels,
                                 uint8_t* output, size_t bytes) {
    if (!input || !output || pixels != 256*256 || bytes < pixels*8) return -1;
    for (size_t i=0; i<pixels; ++i) {
        output[i*8]=input[i];
        for (int j=1;j<8;++j) output[i*8+j]=0;
    }
    return 0;
}
