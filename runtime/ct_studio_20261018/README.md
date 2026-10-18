# CT Studio public runtime assets

Follow the complete delivery guide in docs/CT_Studio完整运行交付_20261018.md.

- ct_studio_runtime_20261018.zip: complete classic + fast host/board packages, synthetic 512x512 CT, frozen original INT8 weights, bit/HWH, drivers and synthetic reference arrays.
- asset_manifest.json: archive/member SHA256 and version identities.
- validation.json: actual software verification and explicitly pending board checks.
- tested-desktop-requirements.txt: exact clean Python 3.11 environment used for preview, optional pip constraints.

Install using `python python/ct_studio_runtime.py install`; never manually bypass nested manifests. Credentials are generated locally, never bundled. The original model, weights and bit bytes are retained. Startup input/reference arrays are NEW synthetic fixtures; old real-CT board acceptance does not certify these new fixtures or unified switching.
