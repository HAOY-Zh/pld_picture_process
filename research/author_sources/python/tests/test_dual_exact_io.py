"""Numerical/layout equality, range guards, bounded cache and fail-closed protocol."""
import ast
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
import zlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'data/CT_Studio_DualExact_20261026_02'
PACKAGE=APP/'data/ct_studio_integration_20261025'
BOARD=PACKAGE/'ct_studio'
sys.path[:0]=[str(APP/'python'),str(BOARD)]
from dual_io_contract import PlaneCache,digest,inflate_exact


class Tests(unittest.TestCase):
    def test_cache_reconstruction_reuse_and_reject_before_mutation(self):
        cache=PlaneCache();q=np.random.default_rng(28).integers(0,256,(5,128,128),dtype=np.uint8)
        def request(q,indices,session='a'*32):
            raw=q.tobytes();hs=[digest(p.tobytes())for p in q]
            return cache.reconstruct(session,hs,indices,b''.join(q[i].tobytes()for i in indices),128*128,digest(raw))
        self.assertEqual(request(q,list(range(5))),q.tobytes())
        q2=np.concatenate([q[1:],q[:1]])
        self.assertEqual(request(q2,[]),q2.tobytes())
        q3=q2.copy();q3[-1]=99
        self.assertEqual(request(q3,[4]),q3.tobytes())
        before=dict(cache.planes)
        with self.assertRaises(ValueError):request(np.full_like(q,78),[])
        self.assertEqual(dict(cache.planes),before)
        with self.assertRaises(ValueError):request(q3,[],session='b'*32)
        self.assertEqual(cache.session,'a'*32)
        with self.assertRaises(ValueError):request(q3,[1,1])
        for i in range(20):
            fresh=np.full_like(q,i);request(fresh,list(range(5)))
            self.assertLessEqual(len(cache.keys()),10)
            self.assertLessEqual(sum(map(len,cache.planes.values())),8*1024*1024)

    def test_empty_bounded_inflation_and_corrupt_stream_rejected(self):
        for size in (0,1,65536):
            packed=zlib.compress(bytes(size));self.assertEqual(inflate_exact(packed,size),bytes(size))
            for bad in (packed[:-1],packed+b'x',zlib.compress(bytes(size+1))):
                with self.assertRaises((ValueError,zlib.error)):inflate_exact(bad,size)

    def test_scheduler_commands_checks_unchanged(self):
        old=(ROOT/'pynq/nodule_canvas_packed.py').read_text();old=old[old.index('    def predict_quantized('):]
        new=(ROOT/'pynq/exact_io_runtime.py').read_text();new=new[new.index('    def _predict_exact('):new.index('\n\nclass ExactUNetPL')]
        new=new.replace('def _predict_exact(', 'def predict_quantized(')
        new=new.replace('list(image.shape)!=self.exact_input_shape','image.shape!=(5,self.height,self.width)')
        new=new.replace('self.exact_packer.pack','self.quantized_packer.pack')
        new=new.replace('                    guard=buf[word_count(self.shapes[name][0]):]\n                    guard.invalidate()\n                    if np.any(np.asarray(guard)!=SENTINEL):',
            '                    buf.invalidate()\n                    if np.any(np.asarray(buf)[word_count(self.shapes[name][0]):]!=SENTINEL):')
        a=new.index('            def read_head():');b=new.index("            if len(commands)",a)
        new=new[:a]+"            codes=timed('output_unpack_ms',lambda:unpack_words(self.buffers['head'],self.shapes['head'][0]))\n"+new[b:]
        self.assertEqual(ast.dump(ast.parse('class X:\n'+old)),ast.dump(ast.parse('class X:\n'+new)))

    def test_segmentation_postprocessing_body_unchanged(self):
        old=(ROOT/'data/CT_Studio_NoduleCompact_20261026_01/python/lung_live_web.py').read_text(encoding='utf8')
        new=(APP/'python/lung_live_web.py').read_text(encoding='utf8')
        a=old.index('            raw, body, plan = native_masks',old.index('    def _run('))
        b=old.index('        if self.cancel.is_set():\n            self._update(phase=\'cancelled\'); return\n        plan = reconstruction_plan',a)
        a2=new.index('            raw, body, plan = native_masks',new.index('    def finish_slice('))
        b2=new.index('\n        # One host result in flight',a2)
        self.assertEqual(old[a:b].strip(),new[a2:b2].strip())

    def test_native_segmentation_layout_and_guard(self):
        from shared_cnn.exact_native_io import ExactPacker
        from shared_cnn.nodule_quantized_pack import compile_pack
        from shared_cnn.lung_axil_smoke import feature_words
        library=compile_pack(BOARD/'shared_cnn/exact_io_native',
            'E:/vivado2/Vivado/2022.2/tps/mingw/6.2.0/win64.o/nt/bin/g++.exe')
        packer=ExactPacker(library,'segmentation')
        for seed in range(3):
            q=np.random.default_rng(seed).integers(0,256,(1,256,256),dtype=np.uint8)
            expected=feature_words(q,0);buf=np.full(len(expected)+16,np.uint64(0xdeadbeef),np.uint64)
            packer.pack(q,buf[:len(expected)])
            np.testing.assert_array_equal(expected,buf[:len(expected)])
            np.testing.assert_array_equal(buf[len(expected):],np.full(16,0xdeadbeef,np.uint64))

    def test_range_sync_preserves_all_tail_checks(self):
        from shared_cnn.exact_io_runtime import ExactSchedule
        from shared_cnn.lung_model_layers_test import SENTINEL
        ranges=[]
        class Buffer(np.ndarray):
            def invalidate(self):ranges.append(self.nbytes)
            def flush(self):pass
        class Packer:
            def pack(self,image,out):out[:]=0
        net=ExactSchedule();net.closed=net.unsafe=net.running=False
        net.exact_input_shape=[1,256,256];net.exact_packer=Packer()
        net.model=dict(input_shape=[1,256,256],nodes=[],commands=0)
        net.shapes={n:([1,256,256],None)for n in ('input','head')}
        net.buffers={n:np.zeros(256*256+16,np.uint64).view(Buffer)for n in net.shapes}
        for buf in net.buffers.values():buf[-16:]=SENTINEL
        net._outstanding=lambda:False;net.cma_bytes=sum(b.nbytes for b in net.buffers.values())
        result,_=net._predict_exact(np.zeros((1,256,256),np.uint8))
        np.testing.assert_array_equal(result,np.zeros_like(result))
        self.assertEqual(ranges,[128,128,256*256*8])
        net.buffers['head'][-1]=0
        with self.assertRaisesRegex(AssertionError,'canary overwritten'):
            net._predict_exact(np.zeros((1,256,256),np.uint8))
        self.assertTrue(net.unsafe)

    def test_binary_cache_http_idle_reconnect_and_fault_no_replay(self):
        from http.server import ThreadingHTTPServer
        import ct_dual_io_service as service_module
        from ct_dual_io_client import CachedCanvasClient,DualSegStudioClient
        bundle=json.loads((PACKAGE/'shared_host_platform.json').read_text())
        class Hardware:
            fault=None;calls=0;fail=False
            manifest=json.loads((BOARD/'platform_manifest.json').read_text())
            def compact(self,q,h,w):
                self.calls+=1
                if self.fail:self.fault='simulated device timeout';raise TimeoutError(self.fault)
                return q[:,::2,::2].copy(),dict(command_count=15,commands=[{}for _ in range(15)],
                    selected_model='nodule_canvas',overlay_reloaded=False,buffer_allocations_per_frame=0,
                    host_total_ms=0.,shared_call_ms=0.,overlay_switch_ms=0.,canvas_setup_ms=0.,pl_command_cycles=0,
                    actual_pl_mhz_before=100.,actual_pl_mhz_after=100.,cpu_mhz=650.)
            def shared_segmentation(self,q):
                return q.copy(),dict(command_count=51,commands=[{}for _ in range(51)],
                    selected_model='unet_original',overlay_reloaded=False,buffer_allocations_per_frame=0,
                    weight_packets_repacked_per_frame=0,host_total_ms=0.,shared_call_ms=0.,
                    overlay_switch_ms=0.,pl_command_cycles=0,
                    clock_before=dict(fclk0_mhz=100.,cpu_mhz=650.),clock_after=dict(fclk0_mhz=100.,cpu_mhz=650.))
        hw=Hardware();service=service_module.Service(hw,'a'*32,bundle['service_bundle_sha256'])
        server=ThreadingHTTPServer(('127.0.0.1',0),service_module.handler_for(service))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=CachedCanvasClient('http://127.0.0.1:'+str(server.server_port),'a'*32,bundle)
        seg=DualSegStudioClient(client.url,'a'*32,bundle)
        try:
            q=np.random.default_rng(7).integers(0,256,(5,128,128),dtype=np.uint8)
            first,r=client.predict(q,128,128);self.assertEqual(r['reused_planes'],0)
            previous=client.io.connection.sock
            second,r=client.predict(q,128,128);self.assertEqual(r['reused_planes'],5)
            self.assertIs(client.io.connection.sock,previous);np.testing.assert_array_equal(first,second)
            client.io.last_success=0
            third,r=client.predict(q,128,128);self.assertIsNot(client.io.connection.sock,previous)
            np.testing.assert_array_equal(third,first)
            codes=np.random.default_rng(8).integers(0,256,(1,256,256),dtype=np.uint8)
            actual,receipt=seg.infer(codes);np.testing.assert_array_equal(actual,codes)
            self.assertEqual(receipt['stats']['command_count'],51)
            hw.fail=True
            with self.assertRaises(RuntimeError):client.predict(q,128,128)
            self.assertEqual(hw.calls,4)
            with self.assertRaises(RuntimeError):client.predict(q,128,128)
            self.assertEqual(hw.calls,4)
        finally:
            client.io.close();seg.io.close();server.shutdown();server.server_close();thread.join(3)

    def test_hardware_receipt_validation_after_completed_canvas(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from ct_dual_io_service import Hardware
        from shared_cnn.exact_io_runtime import ExactCanvasPL
        net=ExactCanvasPL.__new__(ExactCanvasPL);net.height=net.width=128
        net.predict_quantized=lambda q,observer:(np.zeros((5,64,64),np.uint8),dict(
            command_count=15,commands=[{}for _ in range(15)],buffer_allocations_per_frame=0,
            host_total_ms=0.,pl_command_cycles=0))
        owner=SimpleNamespace(_nets={'nodule_canvas':net},_exclusive=nullcontext,_idle=lambda:None,
            _clock_check=lambda:None,_current='nodule_canvas',failed=False)
        hw=Hardware.__new__(Hardware);hw.shared_owner=owner;hw.switch=lambda mode:0.
        hw.folder=BOARD;hw.clocks=SimpleNamespace(fclk0_mhz=100.,cpu_mhz=650.)
        head,stats=hw.compact(np.zeros((5,128,128),np.uint8),128,128)
        self.assertEqual(stats['command_count'],15);self.assertFalse(owner.failed)


if __name__=='__main__':unittest.main()
