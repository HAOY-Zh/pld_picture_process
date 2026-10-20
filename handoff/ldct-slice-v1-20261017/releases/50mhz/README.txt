LDCT slice v1 release: ldct-v1-qeq-safe-50mhz-r3

PRIMARY MODEL: latest epoch118 + validated channel equalization + range-safe INT8/INT16 quantization.
The service owns its hardware lock, Overlay loading, FCLK0 setting, and CMA buffers.
After verified load and clock setup, pass borrowed DMA/GPIO/TX/RX to LDCTProcessor.
The processor never resets, frees, reloads, retries, changes clocks or returns a CPU fallback.

import sys, json
from pathlib import Path
bundle=Path("/absolute/path/to/this/release")
sys.path.insert(0,str(bundle/"pynq"))
from ldct_processor import LDCTProcessor
from pynq import Clocks
manifest=json.loads((bundle/"manifest.json").read_text())
processor=LDCTProcessor(dma=dma,gpio=gpio,tx_buffer=tx_buffer,rx_buffer=rx_buffer,
    model_dir=bundle/"model",read_clock_mhz=lambda:float(Clocks.fclk0_mhz),
    expected_clock_mhz=manifest["clock_mhz"],timeout_s=10.0)
result=processor.process_slice(hu,method="cnn")
processor.assert_idle()
enhanced_hu=result["hu"]

Input dimensions: each 1..512, int16/float32 raw HU; output independent int16, same shape/coordinates.
The service retains spacing/origin/direction. No display window or 0..1 normalization enters this API.
If an exception leaves faulted/dma_may_access true, caller must recover before freeing memory or replacing bit.
Reset/recovery is deliberately absent from LDCTProcessor. Reconstruct it after verified external recovery.

Replay frozen RTL vectors without patient cache:
powershell -ExecutionPolicy Bypass -File source/replay_simulation.ps1 -OutDir ABS_NEW_SIM_DIR
Rebuild this frozen source (new folder, board preset included):
vivado -mode batch -source source/fpga/build_ct_denoise_tile.tcl -tclargs ABS_NEW_BUILD_DIR ABS_RELEASE/source/board_files/pynq-z2 build ABS_RELEASE/source/fpga/ct_denoise_model 50

Host contract tests (NumPy only): python -m unittest discover -s source/tests -p test_ldct_processor.py -v
Owned-resource board benchmark: see pynq/benchmark_ldct_interface.py --help; does not implement an HTTP service.
Record actual readback frequency before/after each call. Never overclock a 50MHz bit to claim a 100MHz release.
