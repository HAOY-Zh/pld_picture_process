#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

struct Axis { size_t lo,hi; double w0,w1; bool valid; };
static Axis axis(double p,unsigned n) {
    Axis a={0,0,0.,0.,false};
    if(!std::isfinite(p)||p<0.||p>double(n-1))return a;
    double f=std::floor(p);a.lo=size_t(f);a.hi=a.lo+1<n?a.lo+1:a.lo;
    a.w0=1.-(p-f);a.w1=1.-a.w0;a.valid=true;return a;
}

// Reproduce order=1 constant-mode ndimage interpolation: double coefficient
// products in y/x order, ordered sum of the four taps, then ONE float32 cast.
// No fast-math/FMA or change of out-of-domain semantics.
API int nodule_resample_xy(const int16_t* src,size_t sc,size_t sy,size_t sx,
    unsigned sh,unsigned sw,unsigned depth,const int32_t* planes,unsigned count,
    const double* ys,unsigned oh,const double* xs,unsigned ow,float* out,int threads) {
    if(!src||!out||!planes||!ys||!xs||!sc||!sy||!sx||!sh||!sw||!depth||!count||!oh||!ow||threads<1||threads>2)return -1;
    for(unsigned k=0;k<count;++k)if(planes[k]<0||unsigned(planes[k])>=depth)return -2;
    std::vector<Axis> y(oh),x(ow);
    for(unsigned i=0;i<oh;++i)y[i]=axis(ys[i],sh);
    for(unsigned i=0;i<ow;++i)x[i]=axis(xs[i],sw);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for(int row=0;row<int(count*oh);++row){
        unsigned k=unsigned(row)/oh,iy=unsigned(row)%oh;
        const int16_t* a=src+size_t(planes[k])*sc;
        float* dst=out+size_t(row)*ow;const Axis& yy=y[iy];
        for(unsigned ix=0;ix<ow;++ix){
            const Axis& xx=x[ix];
            if(!yy.valid||!xx.valid){dst[ix]=-1000.f;continue;}
            double sum=0.,v;
            v=double(a[yy.lo*sy+xx.lo*sx]);v*=yy.w0;v*=xx.w0;sum+=v;
            v=double(a[yy.lo*sy+xx.hi*sx]);v*=yy.w0;v*=xx.w1;sum+=v;
            v=double(a[yy.hi*sy+xx.lo*sx]);v*=yy.w1;v*=xx.w0;sum+=v;
            v=double(a[yy.hi*sy+xx.hi*sx]);v*=yy.w1;v*=xx.w1;sum+=v;
            dst[ix]=float(sum);
        }
    }
    return 0;
}
