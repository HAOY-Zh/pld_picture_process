#include "lung_fp32_kernel.h"
#ifdef LUNG_FP32_NEON_ONLY
extern "C" void lung_fp32_compile_check(const float* x,const float* w,int k,float* out) {lung_fp32_dot4(x,w,k,out);}
#else
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
namespace {
thread_local char error_text[512]={0};
void need(bool ok,const char* s) {if(!ok)throw std::invalid_argument(s);}
void failure(const char* s) {std::strncpy(error_text,s,511);error_text[511]=0;}
struct Tensor {int c,h,w;std::vector<float> data;Tensor(int cc,int hh,int ww):c(cc),h(hh),w(ww),data(size_t(cc)*hh*ww){}};
struct Node {std::array<int32_t,21>d;int padded=0;std::vector<float>weights,bias;};
struct Network {
    int threads;Tensor input;std::vector<Node>nodes;std::vector<Tensor>tensors;
    Network(int c,int h,int w,int t):threads(t),input(c,h,w){}
    const Tensor& source(int i)const{return i<0?input:tensors.at(size_t(i));}
};
void conv(Network& net,size_t index) {
    Node& n=net.nodes[index];const auto& d=n.d;const Tensor& a=net.source(d[1]);Tensor& out=net.tensors[index];
    const int k=d[9],pad=d[10],cin=a.c,cout=out.c,padded=n.padded;
    #pragma omp parallel num_threads(net.threads)
    {
        std::array<float,864>patch{};
        #pragma omp for schedule(static)
        for(int p=0;p<out.h*out.w;++p) {
            const int y=p/out.w,x=p%out.w;int at=0;
            for(int ky=0;ky<k;++ky)for(int kx=0;kx<k;++kx) {
                const int sy=y+ky-pad,sx=x+kx-pad;
                if(sy<0||sx<0||sy>=a.h||sx>=a.w){std::fill(patch.begin()+at,patch.begin()+at+cin,0.f);at+=cin;}
                else {const float* src=&a.data[(size_t(sy)*a.w+sx)*cin];std::copy(src,src+cin,patch.begin()+at);at+=cin;}
            }
            for(int oc=0;oc<cout;oc+=4) {
                float sums[4];lung_fp32_dot4(patch.data(),n.weights.data()+size_t(oc)*padded,padded,sums);
                for(int lane=0;lane<std::min(4,cout-oc);++lane){const int c=oc+lane;float v=sums[lane]+n.bias[c];
                    out.data[size_t(p)*cout+c]=d[13]?std::max(v,0.f):v;}
            }
        }
    }
}
void tensor(Network& net,size_t index) {
    const Node& n=net.nodes[index];const auto& d=n.d;const Tensor& a=net.source(d[1]);Tensor& out=net.tensors[index];
    if(d[0]==2) {
        #pragma omp parallel for num_threads(net.threads) schedule(static)
        for(int p=0;p<out.h*out.w;++p){int y=p/out.w,x=p%out.w;
            const float* r0=&a.data[(size_t(2*y)*a.w+2*x)*a.c];const float* r1=r0+a.c,*r2=r0+a.w*a.c,*r3=r2+a.c;
            for(int c=0;c<a.c;++c)out.data[size_t(p)*a.c+c]=std::max(std::max(r0[c],r1[c]),std::max(r2[c],r3[c]));}
    } else {
        const Tensor& b=net.source(d[2]);
        #pragma omp parallel for num_threads(net.threads) schedule(static)
        for(int p=0;p<out.h*out.w;++p){int y=p/out.w,x=p%out.w;const float* src=&a.data[(size_t(y/2)*a.w+x/2)*a.c];
            float* dst=&out.data[size_t(p)*out.c];std::copy(src,src+a.c,dst);
            src=&b.data[size_t(p)*b.c];std::copy(src,src+b.c,dst+a.c);}
    }
}
}
API const char* lung_fp32_error(){return error_text;}
API int lung_fp32_features(){return LUNG_FP32_NEON|(0
#ifdef _OPENMP
|2
#endif
);}
API void* lung_fp32_create(const int32_t* desc,int count,int c,int h,int w,const float* weights,int nw,const float* bias,int nb,int threads){
    try {
        need(desc&&weights&&bias,"null model");need(count>0&&count<=64&&c>0&&c<=96&&h>0&&h<=512&&w>0&&w<=512,"invalid dimensions");
        need(threads>=1&&threads<=2,"threads must be 1 or 2");
        #ifndef _OPENMP
        need(threads==1,"dual-core needs OpenMP");
        #endif
        std::unique_ptr<Network>net(new Network(c,h,w,threads));size_t storage=net->input.data.size();
        for(int i=0;i<count;++i){Node n;std::copy(desc+i*21,desc+(i+1)*21,n.d.begin());const auto& d=n.d;
            need(d[0]>=1&&d[0]<=3&&d[1]>=-1&&d[1]<i,"invalid dependency/kind");const Tensor& a=net->source(d[1]);
            need(d[3]==a.c&&d[4]==a.h&&d[5]==a.w,"source shape");
            need(d[6]>0&&d[6]<=96&&d[7]>0&&d[7]<=512&&d[8]>0&&d[8]<=512,"output shape");
            if(d[0]==1){need((d[9]==1&&d[10]==0)||(d[9]==3&&d[10]==1),"kernel/padding");
                need(d[7]==a.h&&d[8]==a.w&&(d[13]==0||d[13]==1),"conv shape/relu");int total=a.c*d[9]*d[9],cout=d[6];
                need(d[19]>=0&&int64_t(d[19])+int64_t(total)*cout<=nw&&d[20]>=0&&int64_t(d[20])+cout<=nb,"parameter bounds");
                n.padded=(total+3)/4*4;n.weights.resize(size_t((cout+3)/4*4)*n.padded,0.f);n.bias.assign(bias+d[20],bias+d[20]+cout);
                for(int oc=0;oc<cout;++oc){need(std::isfinite(n.bias[oc]),"nonfinite bias");int j=0;
                    for(int ky=0;ky<d[9];++ky)for(int kx=0;kx<d[9];++kx)for(int ic=0;ic<a.c;++ic){
                        float v=weights[d[19]+((oc*a.c+ic)*d[9]+ky)*d[9]+kx];need(std::isfinite(v),"nonfinite weight");n.weights[size_t(oc)*n.padded+j++]=v;}}
            }else if(d[0]==2){need(a.h%2==0&&a.w%2==0&&d[6]==a.c&&d[7]==a.h/2&&d[8]==a.w/2,"pool shape");}
            else {need(d[2]>=-1&&d[2]<i,"skip dependency");const Tensor& b=net->source(d[2]);
                need(d[6]==a.c+b.c&&d[7]==2*a.h&&d[8]==2*a.w&&b.h==d[7]&&b.w==d[8],"concat shape");}
            storage+=size_t(d[6])*d[7]*d[8];need(storage<=16*1024*1024,"excessive workspace");
            net->nodes.push_back(std::move(n));net->tensors.emplace_back(d[6],d[7],d[8]);
        }error_text[0]=0;return net.release();
    }catch(const std::exception& e){failure(e.what());return nullptr;}
}
API int lung_fp32_forward(void* handle,const float* input,size_t ni,float* output,size_t no){
    try{need(handle&&input&&output,"null forward");Network& net=*static_cast<Network*>(handle);Tensor& a=net.input;
        const Tensor& out=net.tensors.back();need(ni==a.data.size()&&no==out.data.size(),"forward bounds");
        for(int p=0;p<a.h*a.w;++p)for(int c=0;c<a.c;++c)a.data[size_t(p)*a.c+c]=input[size_t(c)*a.h*a.w+p];
        for(size_t i=0;i<net.nodes.size();++i){if(net.nodes[i].d[0]==1)conv(net,i);else tensor(net,i);}
        for(int p=0;p<out.h*out.w;++p)for(int c=0;c<out.c;++c)output[size_t(c)*out.h*out.w+p]=out.data[size_t(p)*out.c+c];return 0;
    }catch(const std::exception& e){failure(e.what());return -1;}
}
API int lung_fp32_trace(void* handle,int i,float* out,size_t count){
    try{need(handle&&out,"null trace");const Network& net=*static_cast<Network*>(handle);need(i>=0&&size_t(i)<net.tensors.size(),"trace index");
        const Tensor& a=net.tensors[i];need(count==a.data.size(),"trace bounds");
        for(int p=0;p<a.h*a.w;++p)for(int c=0;c<a.c;++c)out[size_t(c)*a.h*a.w+p]=a.data[size_t(p)*a.c+c];return 0;
    }catch(const std::exception& e){failure(e.what());return -1;}
}
API void lung_fp32_destroy(void* h){delete static_cast<Network*>(h);}
#endif
