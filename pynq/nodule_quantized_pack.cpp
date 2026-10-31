#include <cstddef>
#include <cstdint>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// Exact original stride-two lowering, fused directly into HWC8 input CMA.
// Source is already quantized. No floating-point arithmetic or new rounding.
API int nodule_pack_quantized(const uint8_t* input, unsigned h, unsigned w,
                             uint8_t* output, size_t bytes) {
    if (!input || !output || h < 128 || w < 128 || h > 1024 || w > 1024 || h%8 || w%8)
        return -1;
    const size_t oh=h/2, ow=w/2, plane=size_t(h)*w;
    if(bytes < oh*ow*48) return -2;
    for(size_t y=0;y<oh;++y) for(size_t x=0;x<ow;++x) {
        uint8_t* dst=output+(y*ow+x)*48;
        for(int c=0;c<5;++c) for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) {
            int yy=int(2*y)+ky-1, xx=int(2*x)+kx-1;
            dst[c*9+ky*3+kx]=(yy<0||xx<0||yy>=int(h)||xx>=int(w))?0:
                input[size_t(c)*plane+size_t(yy)*w+xx];
        }
        dst[45]=dst[46]=dst[47]=0;
    }
    return 0;
}
