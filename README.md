# nvvfx_vs

VapourSynth script that wraps NVIDIA Video Effects (nvidia-vfx) into three clip-processing functions:

| Function | Effect | Output |
|---|---|---|
| `vfx_upscale` | Video Super Resolution (DLSS-based) | RGBS, upscaled resolution |
| `vfx_interpolate` | Video Frame Generation (AI frame interpolation) | RGBS, multiplied frame rate |
| `vfx_truehdr` | TrueHDR (SDR -> HDR10 / BT.2020 / PQ) | RGB30, 10-bit packed RGB |

All three accept any common input format and internally convert to RGBS for processing. Output color/bit-depth conversion can be chained with `core.resize` downstream.

## Requirements

### Hardware
- NVIDIA GPU with Tensor Cores (Turing / Ampere / Ada / Blackwell / Hopper)
- Driver >= 570.65 (Windows) / >= 570.190 (Linux)

### Software
- Python >= 3.10
- VapourSynth (tested on R82)
- nvidia-vfx >= 0.2
- numpy >= 2.0
- PyTorch with CUDA (any recent version compatible with your CUDA toolkit)

### Install dependencies

```bash
pip install nvidia-vfx numpy torch
```

## Installation

Place `nvvfx_vs.py` in your VapourSynth scripts directory:

- Windows: `VapourSynth\Lib\site-packages\vapoursynth\scripts\`
- Linux: `~/.local/lib/python3.x/site-packages/vapoursynth/scripts/`

## Usage

```python
import vapoursynth as vs
core = vs.core

clip = core.ffms2.Source("input.mp4")

# Upscale 2x with ULTRA quality
clip = core.nvvfx_vs.vfx_upscale(clip, factor=2, quality="ULTRA")

# Interpolate to 2x frame rate (e.g. 24fps -> 48fps)
clip = core.nvvfx_vs.vfx_interpolate(clip, factor=2, mode="HIGH")

# Convert SDR to HDR10
clip = core.nvvfx_vs.vfx_truehdr(clip, luminance=1000)

clip.set_output()
```

## API

### vfx_upscale(clip, width=None, height=None, factor=None, quality="ULTRA", strength=1.0, device=0)

Upscale video using NVIDIA Video Super Resolution.

- `width`, `height`: target dimensions (must both be set, or use `factor`)
- `factor`: scale factor (alternative to width/height)
- `quality`: `"BICUBIC" | "LOW" | "MEDIUM" | "HIGH" | "ULTRA"` or `HIGHBITRATE_*`, `DENOISE_*`, `DEBLUR_*`, `STREAMING_*` presets
- `strength`: filter strength [0, 1]
- `device`: CUDA device index

> Recommended upscale ratio <= 4x per pass. For larger ratios, chain multiple calls.

### vfx_interpolate(clip, factor=2, mode="MEDIUM", auto_shot_change=True, device=0)

AI frame interpolation. Output frame count = `(N-1) * factor + 1`, frame rate = `factor * input fps`.

- `factor`: frame rate multiplier (integer >= 2)
- `mode`: `"LOW" | "MEDIUM" | "HIGH"`
- `auto_shot_change`: enable automatic shot change detection (recommended)
- `device`: CUDA device index

### vfx_truehdr(clip, contrast=100, saturation=100, middle_gray=50, luminance=650, debanding_off=0, device=0, set_hdr_props=True)

Convert SDR video to HDR10 (BT.2020 primaries, ST.2084 PQ transfer).

- `contrast`: [0, 200], default 100
- `saturation`: [0, 200], default 100
- `middle_gray`: [10, 100], default 50
- `luminance`: target display peak luminance in nits [400, 2000], default 650
- `debanding_off`: set to 1 to skip deep-learning debanding
- `set_hdr_props`: write BT.2020/PQ/FullRange frame properties (recommended)
- `device`: CUDA device index

Output format is RGB30 (10-bit integer RGB packed as R10G10B10A2).

## Notes

- Effect instances are cached by parameters to avoid reloading AI models.
- A global lock serializes GPU inference since nvvfx objects are not thread-safe for concurrent `run()` calls.
- Performance is currently frame-by-frame synchronous (upload -> infer -> download). Suitable for prototyping and medium resolutions; 4K long-form content would benefit from an async/batched pipeline.

## License

MIT
