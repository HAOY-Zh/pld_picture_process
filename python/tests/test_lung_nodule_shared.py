"""Shared scheduler correctness under independent CPU/device cache emulation.

These are software ownership tests, never FPGA speed or RTL evidence.
"""
import json
from pathlib import Path
import sys
import unittest
import types
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'pynq'),str(ROOT/'python'),str(ROOT/'python/tests')]
from lung_nodule_shared import SharedCNN,load_model
import lung_nodule_shared as shared
from test_lung_fast import CachedMemory,DeviceConv,DeviceTensor
from lung_continuous import unpack_words


class Conv(DeviceConv):
    def _idle(self):
        if self._inflight or self._poisoned:raise RuntimeError('Not idle')


class Tensor(DeviceTensor):
    def _idle(self):
        if self._inflight or self._poisoned:raise RuntimeError('Not idle')


class SharedTests(unittest.TestCase):
    def test_open_shared_downloads_once_and_excludes_second_in_process_owner(self):
        mem=CachedMemory();conv=Conv(mem);conv.mmio=None;tensor=Tensor(mem);loads=[]
        def overlay(bit):
            loads.append(bit)
            return types.SimpleNamespace(lung_conv8_axil_0=types.SimpleNamespace(mmio=None),lung_tensor_axil_0=types.SimpleNamespace(mmio=None))
        clocks=types.SimpleNamespace(fclk0_mhz=100.,cpu_mhz=650.)
        fake=types.SimpleNamespace(Overlay=overlay,allocate=mem.allocate,Clocks=clocks)
        with patch.object(shared,'checked_package',return_value=(Path('.'),{})),patch.dict(sys.modules,{'pynq':fake}),\
            patch.object(shared,'PreparedConv',return_value=conv),patch.object(shared,'TensorAxil',return_value=tensor),\
            patch.object(shared,'compile_io',return_value=(None,{})),patch.object(shared,'NativeLoader',return_value=object()):
            owner,_=shared.open_shared('.')
            try:
                with self.assertRaises(RuntimeError):shared.open_shared('.')
                self.assertEqual(len(loads),1)
                self.assertIs(shared._BOARD_OWNER,owner)
            finally:owner.close()
        self.assertIsNone(shared._BOARD_OWNER)

    def make(self):
        root=ROOT/'data/lung_nodule_shared_20261024_v1';folder=root/'lung_nodule_shared'
        mem=CachedMemory();conv=Conv(mem);tensor=Tensor(mem);owner=SharedCNN(conv,tensor,mem.allocate)
        refs={};images={};models={}
        for label in ['unet_original','unet_pruned','nodule']:
            model,params=load_model(folder,label)
            if label!='nodule':model=json.loads((root/'oracle'/(label+'_32_model.json')).read_text())
            bank=root/'oracle'/('nodule_trace.npz' if label=='nodule' else label+'_32_trace.npz')
            with np.load(bank) as z:refs[label]={n:z[n] for n in z.files}
            images[label]=refs[label]['input'];models[label]=model
            owner.register(label,model,params,'nodule' if label=='nodule' else 'unet')
        return owner,mem,conv,tensor,refs,images,models

    def test_real_frozen_models_interleave_all_nodes_and_preserve_device_payloads(self):
        owner,mem,conv,tensor,refs,images,models=self.make();last={};snapshots={}
        sequence=['unet_original','nodule','unet_pruned','nodule','unet_original']
        initial_allocations=len(mem.buffers)
        try:
            for label in sequence:
                def observe(name,words):np.testing.assert_array_equal(unpack_words(words,models[label]['nodes'][[n['name'] for n in models[label]['nodes']].index(name)]['shape']),refs[label][name])
                head,stats=owner.predict_codes(label,images[label],observe)
                np.testing.assert_array_equal(head,refs[label]['head'])
                self.assertEqual(stats['command_count'],models[label]['commands']);self.assertFalse(stats['overlay_reloaded'])
                self.assertEqual(len(mem.buffers),initial_allocations)
                for other,previous in snapshots.items():
                    if other!=label:
                        for n,a in owner.snapshot(other).items():np.testing.assert_array_equal(a,previous[n])
                snapshots[label]=owner.snapshot(label)
            self.assertTrue(all(b.flushes==1 for net in owner._nets.values() for n,b in net.buffers.items() if n!='input'))
        finally:owner.close()
        self.assertTrue(all(b.freed for b in mem.buffers))

    def test_reentrant_task_or_registration_rejected_without_reloading_or_corrupting(self):
        owner,mem,conv,tensor,refs,images,models=self.make();blocked=[]
        def observe(*a):
            with self.assertRaises(RuntimeError):owner.predict_codes('nodule',images['nodule'])
            with self.assertRaises(RuntimeError):owner.close()
            blocked.append(1)
        try:
            head,_=owner.predict_codes('unet_original',images['unet_original'],observe)
            self.assertEqual(len(blocked),21);np.testing.assert_array_equal(head,refs['unet_original']['head'])
            for label in ['nodule','unet_pruned']:
                head,_=owner.predict_codes(label,images[label]);np.testing.assert_array_equal(head,refs[label]['head'])
        finally:owner.close()

    def test_timeout_prevents_model_switch_and_retains_all_shared_buffers(self):
        owner,mem,conv,tensor,refs,images,models=self.make()
        def fail():conv._poisoned=True;raise TimeoutError('injected uncertain DMA')
        conv.wait=fail
        try:
            with self.assertRaises(TimeoutError):owner.predict_codes('unet_original',images['unet_original'])
            with self.assertRaises(RuntimeError):owner.predict_codes('nodule',images['nodule'])
            with self.assertRaises(RuntimeError):owner.close()
            self.assertFalse(any(b.freed for b in mem.buffers));self.assertTrue(owner.failed)
        finally:
            # Test-double recovery only, never a recommendation for a live board.
            conv._poisoned=conv._inflight=False;owner.close()

    def test_bad_model_label_and_shape_leave_healthy_owner_usable(self):
        owner,mem,conv,tensor,refs,images,models=self.make()
        try:
            with self.assertRaises(ValueError):owner.predict_codes('missing',images['nodule'])
            with self.assertRaises(ValueError):owner.predict_codes('unet_original',images['nodule'])
            head,_=owner.predict_codes('nodule',images['nodule']);np.testing.assert_array_equal(head,refs['nodule']['head'])
            self.assertFalse(owner.failed)
        finally:owner.close()


if __name__=='__main__':unittest.main()
