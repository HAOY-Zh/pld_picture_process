#include "lung_fast_io.h"
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <map>

static void require(bool value) { if (!value) throw std::runtime_error("protocol assertion"); }
struct MockIO {
    std::map<unsigned,uint32_t> reg;
    std::vector<std::pair<unsigned,uint32_t>> writes;
    std::vector<uint32_t> seen,flags;
    bool hung=false,reject=false,drop=false,unloaded=false;
    unsigned ticks=0,polls=0;
    uint32_t read(unsigned address) {
        if (address==8 && ticks && !hung && --ticks==0)
            reg[8]=(reject ? 256u : ((flags.back()&16) && !unloaded ? 8u:32u));
        return reg[address];
    }
    void write(unsigned address,uint32_t value) {
        writes.emplace_back(address,value); reg[address]=value;
        if (address==4 && value==2) reg[8]&=~(2u|4u|256u);
        else if (address==4 && (value&4)) {
            for (unsigned i=0;i<16;++i) seen.push_back(reg[0x100+4*i]);
            flags.push_back(value); if (!drop) ++reg[0x38]; reg[8]=1; ticks=2;
        }
    }
    bool expired() { return ++polls>1024; }
};
int main() {
    try {
        uint32_t params[24]={}; for(unsigned i=8;i<16;++i) params[i]=1073741824;
        for(unsigned i=16;i<24;++i) params[i]=31;
        params[0]=0xffffffffu;
        for (unsigned n : {1u,2u,128u}) {
            std::vector<uint32_t> packets(n*16);
            for(unsigned i=0;i<packets.size();++i) packets[i]=i*0x81234567u;
            MockIO io;io.reg[0x38]=0xfffffffeu;
            require(lung_load_prepared(io,packets.data(),n,params,0x018080ff)==0);
            require(io.seen==packets && io.reg[0x38]==uint32_t(0xfffffffeu+n));
            require(io.writes.size()==26+n*17);
            require(io.writes[0]==std::make_pair(4u,2u));
            require(io.writes[1]==std::make_pair(32u,0x018080ffu));
            for(unsigned i=0;i<24;++i) require(io.writes[2+i]==std::make_pair(0x140+4*i,params[i]));
            for(unsigned i=0;i<n;++i) require(io.flags[i]==(4u | (i==0?8u:0u) | (i+1==n?16u:0u)));
        }
        uint32_t packet[16]={};
        for(unsigned failure=0;failure<5;++failure) {
            MockIO io; io.hung=failure==0; io.reject=failure==1;io.drop=failure==2;io.unloaded=failure==3;
            if(failure==4) io.reg[8]=32;
            const int expected[]={-4,-3,-5,-5,-2};
            require(lung_load_prepared(io,packet,1,params,1)==expected[failure]);
            if(failure==4) require(io.writes.empty());
        }
        for(unsigned count : {0u,129u}) {
            MockIO io;require(lung_load_prepared(io,packet,count,params,1)==-1);require(io.writes.empty());
        }
        MockIO io;params[23]=63;
        require(lung_load_prepared(io,packet,1,params,1)==-1); require(io.writes.empty());
        std::cout<<"native prepared MMIO protocol: PASS (packet order, parameters, wrap, timeout, rejected/incomplete loads)\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
