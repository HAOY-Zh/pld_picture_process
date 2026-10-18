#include "lung_cpu_kernel.h"
#ifdef LUNG_NEON_ONLY
extern "C" void lung_neon_compile_check(const int16_t* x,const int16_t* w,int k,int32_t* s) {lung_dot4(x,w,k,s);}
#else
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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
// Trivial TLS avoids C++ TLS-destructor ordering across a ctypes-loaded DLL.
thread_local char last_error[512]={0};
void set_error(const char* text) {std::strncpy(last_error,text,sizeof(last_error)-1);last_error[sizeof(last_error)-1]=0;}
const int FIELDS=21;
void need(bool ok,const char* text) {if(!ok)throw std::invalid_argument(text);}
uint8_t rq(int32_t acc,int32_t m,int s,int z,bool relu=false) {
    const int64_t product=int64_t(acc)*m;
    const uint64_t magnitude=product<0?uint64_t(-product):uint64_t(product);
    uint64_t value=magnitude>>s;
    if(s) {
        const uint64_t remainder=magnitude&((uint64_t(1)<<s)-1),half=uint64_t(1)<<(s-1);
        value+=(remainder>half || (remainder==half && (value&1)));
    }
    int64_t result=(product<0?-int64_t(value):int64_t(value))+z;
    return uint8_t(std::max<int64_t>(relu?z:0,std::min<int64_t>(255,result)));
}
struct Tensor {
    int c,h,w;std::vector<uint8_t> data;
    Tensor(int cc,int hh,int ww):c(cc),h(hh),w(ww),data(size_t(cc)*hh*ww){}
};
struct Node {
    // kind,a,b,cin,h,w,cout,oh,ow,k,pad,iz,oz,relu,ma,sa,mb,sb,zb,weight_offset,param_offset
    std::array<int32_t,FIELDS> d;
    int padded_k=0;
    std::vector<int16_t> weights;
    std::vector<int32_t> bias,mult,shift;
    std::array<uint8_t,256> lut_a,lut_b;
};
struct Network {
    int threads;Tensor input;
    std::vector<Node> nodes;
    std::vector<Tensor> tensors;
    Network(int c,int h,int w,int t):threads(t),input(c,h,w){}
    const Tensor& source(int i)const {return i<0?input:tensors.at(size_t(i));}
};

void convolution(Network& net,size_t index) {
    Node& n=net.nodes[index];const auto& d=n.d;const Tensor& a=net.source(d[1]);Tensor& out=net.tensors[index];
    const int cin=a.c,cout=out.c,k=d[9],pad=d[10],iz=d[11],oz=d[12],padded=n.padded_k;
    const int pixels=out.h*out.w;
    // Stack-local patch per worker; reused across output channels, no per-pixel heap allocations.
    #pragma omp parallel num_threads(net.threads)
    {
        std::array<int16_t,864> patch{};
        #pragma omp for schedule(static)
        for(int p=0;p<pixels;++p) {
            const int y=p/out.w,x=p%out.w;int at=0;
            for(int ky=0;ky<k;++ky)for(int kx=0;kx<k;++kx) {
                const int sy=y+ky-pad,sx=x+kx-pad;
                if(sy<0||sx<0||sy>=a.h||sx>=a.w) {
                    std::fill(patch.begin()+at,patch.begin()+at+cin,0);at+=cin;
                } else {
                    const uint8_t* src=&a.data[(size_t(sy)*a.w+sx)*cin];
                    for(int c=0;c<cin;++c)patch[at++]=int16_t(int(src[c])-iz);
                }
            }
            // Padding beyond K remains zero; it is initialized once per thread.
            for(int oc=0;oc<cout;oc+=4) {
                int32_t sums[4];lung_dot4(patch.data(),n.weights.data()+size_t(oc)*padded,padded,sums);
                for(int lane=0;lane<std::min(4,cout-oc);++lane) {
                    const int c=oc+lane;
                    out.data[size_t(p)*cout+c]=rq(sums[lane]+n.bias[c],n.mult[c],n.shift[c],oz,d[13]!=0);
                }
            }
        }
    }
}
void tensor_op(Network& net,size_t index) {
    const Node& n=net.nodes[index];const auto& d=n.d;const Tensor& a=net.source(d[1]);Tensor& out=net.tensors[index];
    const int pixels=out.h*out.w;
    if(d[0]==2) {
        #pragma omp parallel for num_threads(net.threads) schedule(static)
        for(int p=0;p<pixels;++p) {
            const int y=p/out.w,x=p%out.w;
            const uint8_t* r0=&a.data[(size_t(2*y)*a.w+2*x)*a.c];
            const uint8_t* r1=r0+a.c;const uint8_t* r2=r0+a.w*a.c;const uint8_t* r3=r2+a.c;
            for(int c=0;c<a.c;++c)out.data[size_t(p)*a.c+c]=std::max(std::max(r0[c],r1[c]),std::max(r2[c],r3[c]));
        }
    } else {
        const Tensor& b=net.source(d[2]);
        #pragma omp parallel for num_threads(net.threads) schedule(static)
        for(int p=0;p<pixels;++p) {
            const int y=p/out.w,x=p%out.w;
            const uint8_t* src=&a.data[(size_t(y/2)*a.w+x/2)*a.c];uint8_t* dst=&out.data[size_t(p)*out.c];
            for(int c=0;c<a.c;++c)dst[c]=n.lut_a[src[c]];
            src=&b.data[size_t(p)*b.c];
            for(int c=0;c<b.c;++c)dst[a.c+c]=n.lut_b[src[c]];
        }
    }
}
}

API const char* lung_cpu_error() {return last_error;}
API int lung_cpu_features() {return LUNG_HAS_NEON | (0
#ifdef _OPENMP
 | 2
#endif
 );}
API uint8_t lung_cpu_requant(int32_t a,int32_t m,int s,int z,int relu) {
    if(m<=0||s<0||s>62||z<0||z>255)return 0;
    return rq(a,m,s,z,relu!=0);
}
API void* lung_cpu_create(const int32_t* desc,int count,int c,int h,int w,
                         const int8_t* weights,int weight_count,const int32_t* bias,
                         const int32_t* mult,const int32_t* shift,int param_count,int threads) {
    try {
        need(desc&&weights&&bias&&mult&&shift,"null model pointer");
        need(count>0&&count<=64&&c>0&&c<=96&&h>0&&h<=512&&w>0&&w<=512,"invalid model dimensions");
        need(threads>=1&&threads<=2,"only one/two CPU worker threads supported");
        #ifndef _OPENMP
        need(threads==1,"dual-core benchmark needs OpenMP build");
        #endif
        std::unique_ptr<Network> net(new Network(c,h,w,threads));size_t storage=net->input.data.size();
        for(int i=0;i<count;++i) {
            Node node;std::copy(desc+i*FIELDS,desc+(i+1)*FIELDS,node.d.begin());const auto& d=node.d;
            need(d[0]>=1&&d[0]<=3&&d[1]>=-1&&d[1]<i,"invalid kind/dependency");
            const Tensor& a=net->source(d[1]);
            need(d[3]==a.c&&d[4]==a.h&&d[5]==a.w,"source shape mismatch");
            need(d[6]>0&&d[6]<=96&&d[7]>0&&d[7]<=512&&d[8]>0&&d[8]<=512,"invalid output geometry");
            need(d[11]>=0&&d[11]<=255&&d[12]>=0&&d[12]<=255,"invalid quantization zero");
            if(d[0]==1) {
                need((d[9]==1&&d[10]==0)||(d[9]==3&&d[10]==1),"invalid kernel/padding");
                need(d[7]==a.h&&d[8]==a.w&&(d[13]==0||d[13]==1),"invalid convolution geometry");
                const int k=d[9],cin=a.c,cout=d[6],total=cin*k*k;
                need(d[19]>=0&&int64_t(d[19])+int64_t(total)*cout<=weight_count,"weight bounds");
                need(d[20]>=0&&int64_t(d[20])+cout<=param_count,"parameter bounds");
                node.padded_k=(total+7)/8*8;
                node.weights.resize(size_t((cout+3)/4*4)*node.padded_k,0);
                node.bias.assign(bias+d[20],bias+d[20]+cout);node.mult.assign(mult+d[20],mult+d[20]+cout);
                node.shift.assign(shift+d[20],shift+d[20]+cout);
                const int8_t* src=weights+d[19];
                for(int oc=0;oc<cout;++oc) {
                    int j=0;int64_t bound=std::abs(int64_t(node.bias[oc]));
                    need(node.mult[oc]>0&&node.shift[oc]>=0&&node.shift[oc]<=62,"invalid Q31 parameters");
                    for(int ky=0;ky<k;++ky)for(int kx=0;kx<k;++kx)for(int ic=0;ic<cin;++ic) {
                        const int16_t v=src[((oc*cin+ic)*k+ky)*k+kx];
                        node.weights[size_t(oc)*node.padded_k+j++]=v;
                        bound+=std::abs(int(v))*std::max(d[11],255-d[11]);
                    }
                    need(bound<std::numeric_limits<int32_t>::max(),"unsafe ACC32 bound");
                }
            } else if(d[0]==2) {
                need(a.h%2==0&&a.w%2==0&&d[6]==a.c&&d[7]==a.h/2&&d[8]==a.w/2&&d[11]==d[12],"pool mismatch");
            } else {
                need(d[2]>=-1&&d[2]<i,"invalid skip source");const Tensor& b=net->source(d[2]);
                need(d[6]==a.c+b.c&&d[7]==a.h*2&&d[8]==a.w*2&&b.h==d[7]&&b.w==d[8],"concat mismatch");
                need(d[14]>0&&d[16]>0&&d[15]>=0&&d[15]<=62&&d[17]>=0&&d[17]<=62&&d[18]>=0&&d[18]<=255,"concat quantization");
                for(int q=0;q<256;++q) {
                    node.lut_a[q]=rq(q-d[11],d[14],d[15],d[12]);
                    node.lut_b[q]=rq(q-d[18],d[16],d[17],d[12]);
                }
            }
            storage+=size_t(d[6])*d[7]*d[8];need(storage<=64*1024*1024,"excessive CPU workspace");
            net->nodes.push_back(std::move(node));net->tensors.emplace_back(d[6],d[7],d[8]);
        }
        last_error[0]=0;return net.release();
    } catch(const std::exception& e) {set_error(e.what());return nullptr;}
}
API int lung_cpu_forward(void* handle,const uint8_t* chw,size_t input_count,uint8_t* output,size_t output_count) {
    try {
        need(handle&&chw&&output,"null forward argument");Network& net=*static_cast<Network*>(handle);
        Tensor& a=net.input;const Tensor& result=net.tensors.back();
        need(input_count==a.data.size()&&output_count==result.data.size(),"wrong forward buffer size");
        for(int p=0;p<a.h*a.w;++p)for(int c=0;c<a.c;++c)a.data[size_t(p)*a.c+c]=chw[size_t(c)*a.h*a.w+p];
        for(size_t i=0;i<net.nodes.size();++i) {
            if(net.nodes[i].d[0]==1)convolution(net,i);else tensor_op(net,i);
        }
        for(int p=0;p<result.h*result.w;++p)for(int c=0;c<result.c;++c)
            output[size_t(c)*result.h*result.w+p]=result.data[size_t(p)*result.c+c];
        return 0;
    } catch(const std::exception& e) {set_error(e.what());return -1;}
}
API int lung_cpu_copy_node(void* handle,int index,uint8_t* chw,size_t count) {
    try {
        need(handle&&chw,"null trace argument");const Network& net=*static_cast<Network*>(handle);
        need(index>=0&&size_t(index)<net.tensors.size(),"invalid trace index");const Tensor& a=net.tensors[index];
        need(count==a.data.size(),"invalid trace length");
        for(int p=0;p<a.h*a.w;++p)for(int c=0;c<a.c;++c)chw[size_t(c)*a.h*a.w+p]=a.data[size_t(p)*a.c+c];
        return 0;
    } catch(const std::exception& e) {set_error(e.what());return -1;}
}
API void lung_cpu_destroy(void* handle) {delete static_cast<Network*>(handle);}
#endif
