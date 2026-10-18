#ifndef LUNG_FAST_IO_H
#define LUNG_FAST_IO_H
#include <cstdint>

// IO is either ordered volatile MMIO or the scripted PC test bus. No arithmetic
// or activations are computed here. Keep the ABI-1 weight push protocol intact.
template<class IO> int lung_load_prepared(IO& io, const uint32_t* packets,
        unsigned count, const uint32_t* parameters, uint32_t quant) {
    if (!packets || !parameters || count < 1 || count > 128 || !(quant & 255)
            || (quant & 0xfe000000u)) return -1;
    for (unsigned lane=0; lane<8; ++lane)
        if (parameters[8+lane] > 0x7fffffffu || parameters[16+lane] > 62) return -1;
    if (io.read(8) & (1|32)) return -2;
    io.write(4,2); io.write(32,quant);
    for (unsigned i=0; i<24; ++i) io.write(0x140+4*i,parameters[i]);
    const uint32_t before=io.read(0x38);
    uint32_t status=0;
    for (unsigned i=0; i<count; ++i) {
        for (unsigned j=0; j<16; ++j) io.write(0x100+4*j,packets[i*16+j]);
        io.write(4,4 | (i==0 ? 8:0) | (i+1==count ? 16:0));
        do {
            status=io.read(8);
            if (status & 256) return -3;
            if (io.expired()) return -4;
        } while (status & 1);
    }
    if (!(status & 8) || (status & 32) || uint32_t(io.read(0x38)-before)!=count) return -5;
    return 0;
}
#endif
