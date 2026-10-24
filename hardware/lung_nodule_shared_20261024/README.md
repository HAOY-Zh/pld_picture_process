# 100 MHz shared U-Net / nodule hardware snapshot

## Current CT Studio integration (2026-10-24)

The batch transport revision uses the same bit/HWH, weights and original
150-file legacy package. Only service framing and PC scheduling change:
`POST /v1/shared/nodule-batch` accepts 1..8 lowered patches and executes them
sequentially under the sole-owner gate. Every patch retains its own hashes,
15-command receipt and fault checks. One PC worker prepares at most one batch
ahead. Completed lung-mask volumes are cached for status refreshes, avoiding
repeated whole-volume scans under the progress lock. Acceptance evidence is in
`data/nodule_speed_20261024_01/`; the pinned selected package is recorded in
`configs/nodule_primary_runtime.json` after acceptance. The original single-patch
route remains supported. Numerical kernels and detector parameters are unchanged.

Same-case physical comparison: 5,237 patches and all 8,288 candidates are exact.
End-to-end detection took 696.98 s before and 431.29 s after (38.12% less time,
1.616x throughput for this single case). Progress callback/lock waiting fell from
256.53 s to 24.55 s; HTTP requests fell from 5,237 to 655. Pure PL execution stayed
about 71.76 s. These are end-to-end orchestration gains, not a faster neural
kernel or detection-quality acceptance. State snapshot measurement on the same
386-slice cleaned mask fell from 37.45 ms per repeated scan to 0.003 ms for the
cached value, with exactly the same volume.

The frozen nodule Verilog core is now installed in the current project's single
hardware owner. This integration reuses the reviewed bit/HWH; it does not modify
RTL, retrain weights or claim a newly rebuilt bit. All 17 RTL files are preserved
here; reviewed artifacts are also in `deploy/ct_nodule_shared.bit/.hwh`.

- Current batch service: `ct-studio-shared-batch.service`, port 8766; bundle at
  `/home/xilinx/ct_test/ct_studio_shared_batch_20261024_01`. The previous
  `ct-studio-shared.service` and its package are retained for rollback.
- Owner: `pynq/ct_studio_shared_service.py`; frozen drivers are packaged under a
  private `shared_cnn` namespace so they cannot replace old segmentation modules.
- Existing web/ROI/LDCT clients retain their separately validated legacy contract.
  The old 150 files, current CNN weights and original lung model are byte-exact.
  The running web's 386-slice case, full lung reconstruction and ROI record remain.
- New authenticated routes: `GET /v1/shared/platform` and
  `POST /v1/shared/nodule-patch`. Input is lowered uint8 `[45,64,64]`, output is the
  actual uint8 `[5,64,64]` head. A patch receipt is explicitly not whole-case detection.
- Independent board audit: 10 calls, 130 node arrays exact (original U-Net and
  four synthetic nodule inputs, each repeated twice). Legacy switching also passed.
- Real CT: 3 distinct patches, 19 hardware calls; head code error 0 against the
  supplied independent FP64/integer reference. Median call 44.46 ms, PL active
  13.70 ms; sampling, transport and candidate postprocessing are separate.
- Existing CNN after switching: 0 HU error against the frozen integer reference.
  Original lung head exactly matches the prior whole-case run; basic and hardware
  ROI output checks passed. 8 new host tests and 40 legacy regression tests passed.
- Evidence index: `data/ct_studio_shared_20261024_01/shared_integration.json`.
  Original startup arrays and independent audit are under `board_startup_proof/`.
  `integration_provenance.json` records original source hashes and the reused bit.
- Whole-case raw-CNN experiment is now connected to the web run/cancel routes.
  The current 386-slice enhanced CT completed 5,237 actual FPGA patch calls and
  produced 8,288 merged raw candidates; candidate `cnn_00001` was linked to MPR,
  3D and a 21-slice hardware ROI. HU moment statistics matched an independent
  calculation exactly. Current HU selection is preserved, including enhanced
  input explicitly marked unvalidated. Final ranking and detection quality are
  still pending; these candidates are not diagnosed nodules.
- Evidence: `data/nodule_online_integration_20261024_01/end_to_end_verification.json`.
  Whole-case elapsed time was 727.5 seconds, separate from per-patch timings.

### Hardware/software boundary

| Part | Current execution |
| --- | --- |
| Existing lightweight denoise CNN and filters | Existing Verilog/PL, unchanged |
| Existing six basic operations | Existing Verilog/PL, unchanged |
| Original lung segmentation | Existing model/Verilog/PL; web contract unchanged |
| ROI filter/count/HU sum/sumsq/min/max | Existing RST1 Verilog/PL |
| Nodule CNN convolutions and tensor operators | Reviewed shared Verilog/PL, now in the same owner |
| HU mean/std from moments, file I/O, scheduling | PC/ARM |
| MPR, local MIP, three-dimensional rendering and selection | PC display path |
| Full-case nodule search/sampling/decode/merge | PC pipeline with actual sole-owner FPGA patch inference; online raw-CNN experiment |
| Final ranker/filter calibration and detection quality | Pending algorithm-owner confirmation and annotated-case acceptance |

The service is enabled for startup after physical acceptance; old units and
packages are retained. No board reboot was performed in this integration.

For an additional real-patch acceptance run, use a fresh output folder and the
explicit `python/verify_ct_studio_shared_board.py` command interface. It submits
actual hardware work; do not run it concurrently with web computations.

The 17 RTL files match the accepted bit SHA-256 `21523a8034acf96b2b4c20615353a9a7804cc0c4f5a6f05f1c1592d020d47187`. The reviewed bit/HWH are delivered in `runtime/lung_nodule_shared_20261024/` through the checked installer. Use them for initial integration.

`build.tcl`, the PS settings, reset validation and XDC were adapted from the accepted build by changing paths only. The portable entry point has not been rerun. The original build required post-route physical optimization with unchanged 100 MHz constraints; `resume_physopt.tcl` provides that step when the initial timing gate fails. Never bypass setup/hold/DRC gates. A rebuild may have a different bit identity and requires new physical acceptance.

See [whole-case integration requirements](../../docs/肺结节自动检测_队友补充交付要求_20261024.md). This version does not include LDCT or basic processing in the shared bit; the owner switches their original overlays only when idle.
