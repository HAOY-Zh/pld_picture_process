#pragma once
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define LUNG_FP32_NEON 1
#else
#define LUNG_FP32_NEON 0
#endif
// Four output channels reuse one input patch. FP32 operations, not INT8 decode.
inline void lung_fp32_dot4(const float* x,const float* w,int k,float* out) {
#if LUNG_FP32_NEON
    float32x4_t a=vdupq_n_f32(0),b=a,c=a,d=a;
    for(int j=0;j<k;j+=4) {
        float32x4_t v=vld1q_f32(x+j);
        a=vmlaq_f32(a,v,vld1q_f32(w+j));b=vmlaq_f32(b,v,vld1q_f32(w+k+j));
        c=vmlaq_f32(c,v,vld1q_f32(w+2*k+j));d=vmlaq_f32(d,v,vld1q_f32(w+3*k+j));
    }
    float32x2_t aa=vadd_f32(vget_low_f32(a),vget_high_f32(a));
    float32x2_t bb=vadd_f32(vget_low_f32(b),vget_high_f32(b));
    float32x2_t cc=vadd_f32(vget_low_f32(c),vget_high_f32(c));
    float32x2_t dd=vadd_f32(vget_low_f32(d),vget_high_f32(d));
    out[0]=vget_lane_f32(vpadd_f32(aa,aa),0);out[1]=vget_lane_f32(vpadd_f32(bb,bb),0);
    out[2]=vget_lane_f32(vpadd_f32(cc,cc),0);out[3]=vget_lane_f32(vpadd_f32(dd,dd),0);
#else
    // Match the ARM four-lane accumulation order for native numeric tests.
    for(int o=0;o<4;++o) {
        float s[4]={0,0,0,0};
        for(int j=0;j<k;j+=4)for(int i=0;i<4;++i)s[i]+=x[j+i]*w[o*k+j+i];
        out[o]=(s[0]+s[2])+(s[1]+s[3]);
    }
#endif
}
