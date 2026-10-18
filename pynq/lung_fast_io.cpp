#include "lung_fast_io.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>

static inline void barrier() {
#if defined(__arm__)
    __asm__ __volatile__("dmb sy" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}
struct MappedIO {
    volatile uint32_t* base;
    std::chrono::steady_clock::time_point deadline;
    uint32_t read(unsigned offset) {
        barrier(); const uint32_t value=base[offset/4]; barrier(); return value;
    }
    void write(unsigned offset,uint32_t value) {
        barrier(); base[offset/4]=value; barrier();
    }
    bool expired() const { return std::chrono::steady_clock::now()>=deadline; }
};
extern "C" {
// Pointer lifetime belongs to Python's PYNQ MMIO mapping. This function never
// maps arbitrary addresses, starts inference, frees buffers, or resets AXI.
#ifdef _WIN32
__declspec(dllexport)
#endif
int lung_fast_load(uint32_t* registers, size_t register_words,
        const uint32_t* packets,unsigned count,const uint32_t* params,
        uint32_t quant,double timeout) {
    if (!registers || (reinterpret_cast<uintptr_t>(registers)&3) || register_words<0x1a0/4
            || !std::isfinite(timeout) || timeout<=0 || timeout>3600) return -1;
    MappedIO io{registers, std::chrono::steady_clock::now()+
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout))};
    return lung_load_prepared(io,packets,count,params,quant);
}
}
