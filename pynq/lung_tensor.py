"""MMIO driver for PL pool/nearest/requantized concat tensor engine, ABI 1."""
import math
import time

from lung_axil import bounded, LungAxil

ID,CTRL,STATUS,VERSION=0,4,8,12
A,B,DST,SHAPE,GEOMETRY,ZEROS=16,20,24,28,32,36
MA,SA,MB,SB,WRITTEN,READS,CYCLES,COMPLETED=40,44,48,52,56,60,64,68
BUSY,DONE,ERROR,CONTROL_ERROR=1,2,4,256


def layout(op,height,width,channels_a,channels_b=0):
    op=bounded(op,0,3,'op');h=bounded(height,1,512,'height');w=bounded(width,1,512,'width')
    ca=bounded(channels_a,1,96,'channels_a');cb=bounded(channels_b,0,96,'channels_b')
    if (op<2 and cb) or (op>=2 and (cb==0 or ca+cb>96)):
        raise ValueError('Invalid concatenation channels')
    if op==0 and (h%2 or w%2): raise ValueError('Pool input must be even')
    if op in (1,3) and max(h,w)>256: raise ValueError('Upsampled output exceeds 512')
    oh,ow=(h//2,w//2) if op==0 else ((h*2,w*2) if op in (1,3) else (h,w))
    cout=ca+cb if op>=2 else ca
    return dict(op=op,height=h,width=w,ca=ca,cb=cb,oh=oh,ow=ow,cout=cout,
                a_bytes=h*w*((ca+7)//8)*8,b_bytes=(oh*ow*((cb+7)//8)*8 if cb else 0),
                dst_bytes=oh*ow*((cout+7)//8)*8,words=oh*ow*((cout+7)//8))


class TensorAxil:
    def __init__(self,mmio,timeout=30.0,poll_interval=0.0):
        if not math.isfinite(timeout) or timeout<=0 or not math.isfinite(poll_interval) or poll_interval<0:
            raise ValueError('Invalid polling settings')
        self.mmio=mmio;self.timeout=float(timeout);self.poll_interval=float(poll_interval)
        self._inflight=self._poisoned=False;self._config=None
        if mmio.read(ID)!=0x54454e53 or mmio.read(VERSION)!=0x10000:
            raise RuntimeError('Wrong tensor ID/ABI; load the new dual-engine bit/HWH')

    def _idle(self):
        if self._poisoned: raise RuntimeError('Shared AXI/board recovery required before reuse')
        status=self.mmio.read(STATUS)
        if self._inflight or status&BUSY: raise RuntimeError('Tensor command still outstanding')

    def configure(self,a,dst,op,height,width,channels_a,b=0,channels_b=0,
                  za=0,zb=0,zo=0,ma=1,sa=0,mb=1,sb=0):
        config=layout(op,height,width,channels_a,channels_b)
        za=bounded(za,0,255,'za');zb=bounded(zb,0,255,'zb');zo=bounded(zo,0,255,'zo')
        ma=bounded(ma,1,2**31-1,'ma');mb=bounded(mb,1,2**31-1,'mb')
        sa=bounded(sa,0,62,'sa');sb=bounded(sb,0,62,'sb')
        if op<2 and za!=zo: raise ValueError('Pool/nearest preserve quantization grid')
        a=bounded(a,0,2**32-1,'a');dst=bounded(dst,0,2**32-1,'dst');b=bounded(b,0,2**32-1,'b')
        for address,size in [(a,config['a_bytes']),(dst,config['dst_bytes'])]+([(b,config['b_bytes'])] if op>=2 else []):
            if address%8 or address+size>2**32: raise ValueError('Unaligned or wrapped buffer range')
        if any(src<dst+config['dst_bytes'] and dst<src+size for src,size in
               [(a,config['a_bytes'])]+([(b,config['b_bytes'])] if op>=2 else [])):
            raise ValueError('Output must not overlap complete input tensors')
        self._idle()
        values=[a,b,dst,config['height']|(config['width']<<16),op|(config['ca']<<8)|(config['cb']<<16),
                za|(zb<<8)|(zo<<16),ma,sa,mb,sb]
        for index,value in enumerate(values): self.mmio.write(16+4*index,value)
        self._config=config|dict(a=a,b=b,dst=dst)

    def start(self):
        self._idle()
        if self._config is None: raise RuntimeError('Configure tensor before START')
        self._inflight=True
        try: self.mmio.write(CTRL,1)
        except BaseException:
            self._poisoned=True
            raise

    def wait(self):
        if self._poisoned or not self._inflight: raise RuntimeError('No healthy outstanding tensor command')
        deadline=time.monotonic()+self.timeout
        while True:
            status=self.mmio.read(STATUS)
            if status&(DONE|CONTROL_ERROR) and not status&BUSY: break
            if time.monotonic()>=deadline:
                self._poisoned=True
                raise TimeoutError('Tensor timeout; retain buffers and recover shared AXI/board')
            if self.poll_interval: time.sleep(self.poll_interval)
        self._inflight=False
        if status&(ERROR|CONTROL_ERROR): raise RuntimeError('Tensor command failed; discard entire output')
        stats={name:self.mmio.read(address) for name,address in [('written_words',WRITTEN),('read_words',READS),
                                                                ('cycles',CYCLES),('completed',COMPLETED)]}
        if stats['written_words']!=self._config['words']: raise RuntimeError('Tensor output count mismatch')
        return stats

    def run(self,src,dst,op,height,width,channels_a,src_b=None,channels_b=0,**quant):
        sizes=layout(op,height,width,channels_a,channels_b)
        buffers=[(src,sizes['a_bytes']),(dst,sizes['dst_bytes'])]
        if op>=2: buffers.append((src_b,sizes['b_bytes']))
        for buffer,size in buffers:
            if buffer is None or buffer.nbytes<size or not callable(getattr(buffer,'flush',None)) or not callable(getattr(buffer,'invalidate',None)):
                raise ValueError('Use sufficiently sized PYNQ device buffers with cache synchronization')
        self.configure(LungAxil._buffer_address(src),LungAxil._buffer_address(dst),op,height,width,channels_a,
                       b=LungAxil._buffer_address(src_b) if op>=2 else 0,channels_b=channels_b,**quant)
        for buffer,_ in buffers: buffer.flush()
        self.start();stats=self.wait();dst.invalidate()
        return stats
