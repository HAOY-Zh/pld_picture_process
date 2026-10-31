#include <cstddef>
#include <cstdint>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// Identical lookup/lowering contract. HWC8 writes whole consecutive pixels,
// avoiding 45 separate strided passes over the destination cache lines.
API int nodule_canvas_pack(const int16_t* image,size_t sc,size_t sy,size_t sx,
    unsigned height,unsigned width,const uint8_t* lut,uint8_t* out,size_t bytes,int hwc8) {
    if(!image||!lut||!out||!sc||!sy||!sx||height<128||width<128||height>1024||width>1024
        ||height%8||width%8||(hwc8!=0&&hwc8!=1))return -1;
    const size_t h=height/2,w=width/2,pixels=h*w,channels=hwc8?48:45;
    if(bytes<pixels*channels)return -2;
    if(hwc8){
        for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){
            uint8_t* dest=out+(y*w+x)*48;
            for(int c=0;c<5;++c)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
                const int iy=int(2*y)+ky-1,ix=int(2*x)+kx-1;
                dest[c*9+ky*3+kx]=(iy<0||ix<0||iy>=int(height)||ix>=int(width))?0:
                    lut[int(image[size_t(c)*sc+size_t(iy)*sy+size_t(ix)*sx])+32768];
            }
            dest[45]=dest[46]=dest[47]=0;
        }
    }else{
        for(int c=0;c<5;++c)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
            uint8_t* dest=out+size_t(c*9+ky*3+kx)*pixels;
            for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){
                const int iy=int(2*y)+ky-1,ix=int(2*x)+kx-1;
                dest[y*w+x]=(iy<0||ix<0||iy>=int(height)||ix>=int(width))?0:
                    lut[int(image[size_t(c)*sc+size_t(iy)*sy+size_t(ix)*sx])+32768];
            }
        }
    }
    return 0;
}
