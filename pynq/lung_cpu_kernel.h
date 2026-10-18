#pragma once
#include <stdint.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define LUNG_HAS_NEON 1
#else
#define LUNG_HAS_NEON 0
#endif

// Four output channels share one input patch. Inputs/weights are signed int16;
// multiply into int32, never int16 accumulation. k is padded to eight.
static inline void lung_dot4(const int16_t* x,const int16_t* w,int k,int32_t* sums) {
#if LUNG_HAS_NEON
    int32x4_t a0=vdupq_n_s32(0),b0=a0,a1=a0,b1=a0,a2=a0,b2=a0,a3=a0,b3=a0;
    for(int j=0;j<k;j+=8) {
        const int16x8_t v=vld1q_s16(x+j);
        int16x8_t q=vld1q_s16(w+j);
        a0=vmlal_s16(a0,vget_low_s16(v),vget_low_s16(q));
        b0=vmlal_s16(b0,vget_high_s16(v),vget_high_s16(q));
        q=vld1q_s16(w+k+j);
        a1=vmlal_s16(a1,vget_low_s16(v),vget_low_s16(q));
        b1=vmlal_s16(b1,vget_high_s16(v),vget_high_s16(q));
        q=vld1q_s16(w+2*k+j);
        a2=vmlal_s16(a2,vget_low_s16(v),vget_low_s16(q));
        b2=vmlal_s16(b2,vget_high_s16(v),vget_high_s16(q));
        q=vld1q_s16(w+3*k+j);
        a3=vmlal_s16(a3,vget_low_s16(v),vget_low_s16(q));
        b3=vmlal_s16(b3,vget_high_s16(v),vget_high_s16(q));
    }
    const int32x4_t vectors[4]={vaddq_s32(a0,b0),vaddq_s32(a1,b1),vaddq_s32(a2,b2),vaddq_s32(a3,b3)};
    for(int i=0;i<4;++i) {
        const int64x2_t p=vpaddlq_s32(vectors[i]);
        sums[i]=(int32_t)(vgetq_lane_s64(p,0)+vgetq_lane_s64(p,1));
    }
#else
    int32_t a=0,b=0,c=0,d=0;
    for(int j=0;j<k;++j) {
        a+=int32_t(x[j])*w[j]; b+=int32_t(x[j])*w[k+j];
        c+=int32_t(x[j])*w[2*k+j]; d+=int32_t(x[j])*w[3*k+j];
    }
    sums[0]=a;sums[1]=b;sums[2]=c;sums[3]=d;
#endif
}
