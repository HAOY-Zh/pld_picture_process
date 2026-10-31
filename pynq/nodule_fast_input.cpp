#include <cstddef>
#include <cstdint>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// Integer data movement only. LUT embodies the unchanged HU quantizer.
// Positive element strides support cropped views of the shared-plane canvas.
API int nodule_pack(const int16_t* image,size_t sc,size_t sy,size_t sx,
                    const uint8_t* lut,uint8_t* out,size_t bytes,int hwc8) {
    const size_t pixels=64*64,channels=hwc8?48:45;
    if(!image||!lut||!out||!sc||!sy||!sx||bytes<pixels*channels||(hwc8!=0&&hwc8!=1))return -1;
    for(int c=0;c<5;++c)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx) {
        const int channel=c*9+ky*3+kx;
        for(int y=0;y<64;++y)for(int x=0;x<64;++x) {
            const int iy=y*2+ky-1,ix=x*2+kx-1;
            uint8_t code=0;
            if(iy>=0&&ix>=0&&iy<128&&ix<128) {
                const int v=image[size_t(c)*sc+size_t(iy)*sy+size_t(ix)*sx];
                code=lut[v+32768];
            }
            const size_t p=size_t(y)*64+x;
            out[hwc8?p*48+channel:size_t(channel)*pixels+p]=code;
        }
    }
    if(hwc8)for(size_t p=0;p<pixels;++p)out[p*48+45]=out[p*48+46]=out[p*48+47]=0;
    return 0;
}
