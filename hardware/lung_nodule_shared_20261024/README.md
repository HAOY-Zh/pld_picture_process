# 100 MHz shared U-Net / nodule hardware snapshot

The 17 RTL files match the accepted bit SHA-256 `21523a8034acf96b2b4c20615353a9a7804cc0c4f5a6f05f1c1592d020d47187`. The reviewed bit/HWH are delivered in `runtime/lung_nodule_shared_20261024/` through the checked installer. Use them for initial integration.

`build.tcl`, the PS settings, reset validation and XDC were adapted from the accepted build by changing paths only. The portable entry point has not been rerun. The original build required post-route physical optimization with unchanged 100 MHz constraints; `resume_physopt.tcl` provides that step when the initial timing gate fails. Never bypass setup/hold/DRC gates. A rebuild may have a different bit identity and requires new physical acceptance.

See [delivery instructions](../../docs/肺分割与结节共享引擎GitHub交付_20261024.md). This version does not include LDCT or basic processing in the shared bit.
