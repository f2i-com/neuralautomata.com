# Local validation — 13 September 2026

This is functional acceptance on Windows with Chrome 152, Python 3.11,
PyTorch 2.11.0+cu128, CUDA 12.8 and two NVIDIA GeForce RTX 5090 devices.
It is not a throughput benchmark or a portability claim for untested hardware.

- Native bridge: 6 Node tests passed, including UTF-8 JSON with charset,
  device UUID mapping, launch failures and request validation.
- Imported research: 22 pytest cases passed from the repository root.
- `python tests/smoke.py`: all four experiments passed on both CUDA devices;
  explicit CPU, stop/reload, Unicode generation, telemetry and lost WebGL context passed.
- Fresh memory/language captures contain 60/72 frames and infinite GIF loops.
- All 97 research source/checkpoint/result files match `research-import.json`.

The original research result files describe their historical CPU baseline.
The later GPU runs validate the migrated application and do not replace those results.
