from __future__ import annotations

import ctypes
import threading
import traceback
import warnings

import vapoursynth as vs
import numpy as np

core = vs.core

__version__ = "0.2.0"
__all__ = ["vfx_upscale", "vfx_interpolate", "vfx_truehdr", "to_rgbs"]

try:
    from nvvfx import VideoSuperRes, VideoFrameGeneration, TrueHDR
    _NV_VFX_OK = True
except ImportError:
    VideoSuperRes = VideoFrameGeneration = TrueHDR = None
    _NV_VFX_OK = False

_EFFECTS: dict = {}
_EFFECT_LOCK = threading.Lock()
_local = threading.local()


def _require_nvvfx() -> None:
    if not _NV_VFX_OK:
        raise RuntimeError(
            "nvidia-vfx is not installed. Run: pip install nvidia-vfx"
        )


def _require_torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch with CUDA is required: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in PyTorch.")
    return torch


def _resolve_enum(enum_cls, value, name: str):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        v = value.strip().upper()
        if hasattr(enum_cls, v):
            return getattr(enum_cls, v)
        raise vs.Error(f"nvvfx_vs: unknown {name}={value!r}, choices: {[e.name for e in enum_cls]}")
    raise vs.Error(f"nvvfx_vs: {name} expects enum or str, got {type(value).__name__}")


def _get_effect(cls, key, factory):
    with _EFFECT_LOCK:
        eff = _EFFECTS.get((cls, key))
        if eff is None:
            eff = factory()
            _EFFECTS[(cls, key)] = eff
        return eff


_FORMATS: dict = {}


def _get_formats():
    if not _FORMATS:
        _FORMATS["rgbs"] = core.query_video_format(vs.RGB, vs.FLOAT, 32, 0, 0)
        _FORMATS["rgb30"] = core.query_video_format(vs.RGB, vs.INTEGER, 10, 0, 0)
    return _FORMATS["rgbs"], _FORMATS["rgb30"]


class _Buf:
    __slots__ = ("dev", "h", "w", "n_in", "cuda_in", "stage", "stage2")

    def __init__(self, torch, dev, h, w, n_in):
        self.dev = dev
        self.h, self.w = h, w
        self.n_in = n_in
        self.cuda_in = [torch.empty(3, h, w, dtype=torch.float32, device=dev)
                        for _ in range(n_in)]
        self.stage = np.empty((3, h, w), dtype=np.float32)
        self.stage2 = np.empty((3, h, w), dtype=np.float32) if n_in == 2 else None


def _get_buf(torch, dev, h, w, n_in):
    key = (dev, h, w, n_in)
    b = getattr(_local, "buf", None)
    if b is None or (b.dev, b.h, b.w, b.n_in) != key:
        b = _Buf(torch, dev, h, w, n_in)
        _local.buf = b
    return b


def _read_into(stage, frame):
    w, h = frame.width, frame.height
    for p in range(3):
        stride = frame.get_stride(p)
        addr = frame.get_read_ptr(p).value
        buf = (ctypes.c_uint8 * (h * stride)).from_address(addr)
        raw = np.ctypeslib.as_array(buf).reshape(h, stride)
        plane_view = raw[:, :w * 4].reshape(h, w, 4).view(np.float32).reshape(h, w)
        np.copyto(stage[p], plane_view)


def _write_plane(frame, plane, data):
    h, w = data.shape
    itemsize = data.dtype.itemsize
    stride = frame.get_stride(plane)
    addr = frame.get_write_ptr(plane).value
    buf = (ctypes.c_uint8 * (h * stride)).from_address(addr)
    raw = np.ctypeslib.as_array(buf).reshape(h, stride)
    row_bytes = w * itemsize
    flat = data.reshape(h, w, 1).view(np.uint8).reshape(h, row_bytes)
    raw[:, :row_bytes] = flat


def _write_rgb_numpy(dst, rgb):
    for p in range(3):
        _write_plane(dst, p, rgb[p])


def _frame_at(f, idx):
    if isinstance(f, (list, tuple)):
        return f[idx]
    return f


def _ev_debug(fn):
    def wrapper(n, f):
        try:
            return fn(n, f)
        except BaseException:
            traceback.print_exc()
            raise
    return wrapper


def _run_and_clone(torch, dev, run_call):
    stream = torch.cuda.current_stream(device=dev)
    result = run_call(stream.cuda_stream)
    torch.cuda.synchronize(device=dev)
    out = torch.from_dlpack(result.image).clone()
    stream.synchronize()
    return out


def to_rgbs(clip, matrix_in_s="709", range_in_s="limited"):
    fmt = clip.format
    if fmt is None:
        raise vs.Error("nvvfx_vs: variable-format input is not supported")
    if fmt.color_family == vs.RGB:
        if fmt.sample_type == vs.FLOAT and fmt.bits_per_sample == 32:
            return clip
        return core.resize.Bicubic(clip, format=vs.RGBS)
    if fmt.color_family == vs.GRAY:
        return core.resize.Bicubic(clip, format=vs.RGBS)
    return core.resize.Bicubic(
        clip, format=vs.RGBS,
        matrix_in_s=matrix_in_s,
        range_in_s=range_in_s, range_s="full",
    )


def vfx_upscale(clip, width=None, height=None, factor=None,
                quality="ULTRA", strength=1.0, device=0):
    _require_nvvfx()
    src = to_rgbs(clip)
    W, H = src.width, src.height

    if (width is None) != (height is None):
        raise vs.Error("vfx_upscale: width and height must be given together, or use factor")
    if width is None:
        if factor is None:
            raise vs.Error("vfx_upscale: provide width/height or factor")
        width, height = W * factor, H * factor
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise vs.Error("vfx_upscale: output dimensions must be positive even integers")

    if max(width / W, height / H) > 4:
        warnings.warn(
            "vfx_upscale: upscale ratio exceeds 4x; NVIDIA recommends <=4x. "
            "Consider chaining two passes (e.g. 2x then 2x).",
            UserWarning, stacklevel=2
        )

    q = _resolve_enum(VideoSuperRes.QualityLevel, quality, "quality")
    key = ("sr", device, int(width), int(height), q, float(strength))

    def factory():
        sr = VideoSuperRes(quality=q, strength=strength, device=device)
        sr.output_width = int(width)
        sr.output_height = int(height)
        sr.load()
        return sr

    sr = _get_effect(VideoSuperRes, key, factory)
    torch = _require_torch()
    rgbs, _ = _get_formats()
    dev = f"cuda:{device}"

    def ev(n, f):
        buf = _get_buf(torch, dev, H, W, 1)
        _read_into(buf.stage, _frame_at(f, 0))
        buf.cuda_in[0].copy_(torch.from_numpy(buf.stage), non_blocking=True)
        with _EFFECT_LOCK:
            out_img = _run_and_clone(torch, dev,
                lambda sp: sr.run(buf.cuda_in[0], stream_ptr=sp, non_blocking=False))
        arr = out_img.detach().cpu().numpy()
        dst = core.create_video_frame(rgbs.id, int(width), int(height))
        _write_rgb_numpy(dst, arr)
        return dst

    out = core.std.BlankClip(clip=src, width=int(width), height=int(height),
                             format=rgbs.id)
    return core.std.ModifyFrame(clip=out, clips=[src], selector=_ev_debug(ev))


def vfx_interpolate(clip, factor=2, mode="MEDIUM", auto_shot_change=True, device=0):
    _require_nvvfx()
    if factor < 2 or int(factor) != factor:
        raise vs.Error("vfx_interpolate: factor must be an integer >= 2")
    factor = int(factor)

    src = to_rgbs(clip)
    L = src.num_frames
    if L < 2:
        raise vs.Error("vfx_interpolate: at least 2 frames required")

    m = _resolve_enum(VideoFrameGeneration.Mode, mode, "mode")
    key = ("vfg", device, m, src.width, src.height, bool(auto_shot_change))

    def factory():
        vfg = VideoFrameGeneration(
            mode=m,
            automatic_shot_change_detection_enabled=bool(auto_shot_change),
            device=device,
        )
        vfg.input_width = src.width
        vfg.input_height = src.height
        vfg.frame_multiplier = factor
        vfg.load()
        return vfg

    vfg = _get_effect(VideoFrameGeneration, key, factory)
    torch = _require_torch()
    rgbs, _ = _get_formats()
    dev = f"cuda:{device}"
    W, H = src.width, src.height

    out_len = (L - 1) * factor + 1

    def _dup(clip_):
        frames = [i for i in range(clip_.num_frames) for _ in range(factor - 1)]
        return core.std.DuplicateFrames(clip_, frames=frames)

    A = _dup(core.std.Trim(src, first=0, last=L - 2))
    B = _dup(core.std.Trim(src, first=1, last=L - 1))
    last = core.std.Trim(src, first=L - 1, last=L - 1)
    A = A + last
    B = B + last

    def ev(n, f):
        j = n % factor
        if j == 0:
            return _frame_at(f, 0).copy()
        buf = _get_buf(torch, dev, H, W, 2)
        _read_into(buf.stage, _frame_at(f, 0))
        _read_into(buf.stage2, _frame_at(f, 1))
        buf.cuda_in[0].copy_(torch.from_numpy(buf.stage), non_blocking=True)
        buf.cuda_in[1].copy_(torch.from_numpy(buf.stage2), non_blocking=True)
        with _EFFECT_LOCK:
            out_img = _run_and_clone(torch, dev,
                lambda sp: vfg.run(buf.cuda_in[0], buf.cuda_in[1],
                                   frame_index=j, stream_ptr=sp, non_blocking=False))
        arr = out_img.detach().cpu().numpy()
        dst = core.create_video_frame(rgbs.id, W, H)
        _write_rgb_numpy(dst, arr)
        return dst

    fps_num = src.fps_num * factor
    fps_den = src.fps_den
    out = core.std.BlankClip(clip=src, length=out_len,
                             fpsnum=fps_num, fpsden=fps_den, format=rgbs.id)
    return core.std.ModifyFrame(clip=out, clips=[A, B], selector=_ev_debug(ev))


def vfx_truehdr(clip, contrast=100, saturation=100, middle_gray=50,
                luminance=650, debanding_off=0, device=0, set_hdr_props=True):
    _require_nvvfx()
    src = to_rgbs(clip)
    W, H = src.width, src.height

    key = ("thdr", device, int(contrast), int(saturation), int(middle_gray),
           int(luminance), int(debanding_off))

    def factory():
        return TrueHDR(contrast=int(contrast), saturation=int(saturation),
                       middle_gray=int(middle_gray), luminance=int(luminance),
                       debanding_off=int(debanding_off), device=device)

    thdr = _get_effect(TrueHDR, key, factory)
    torch = _require_torch()
    _, rgb30 = _get_formats()
    dev = f"cuda:{device}"

    def ev(n, f):
        buf = _get_buf(torch, dev, H, W, 1)
        _read_into(buf.stage, _frame_at(f, 0))
        buf.cuda_in[0].copy_(torch.from_numpy(buf.stage), non_blocking=True)
        with _EFFECT_LOCK:
            packed = _run_and_clone(torch, dev,
                lambda sp: thdr.run(buf.cuda_in[0], stream_ptr=sp, non_blocking=False))
        px = packed.detach().cpu().numpy().astype(np.uint32)
        r = (px & 0x3FF).astype(np.uint16)
        g = ((px >> 10) & 0x3FF).astype(np.uint16)
        b = ((px >> 20) & 0x3FF).astype(np.uint16)

        dst = core.create_video_frame(rgb30.id, W, H)
        _write_rgb_numpy(dst, np.stack([r, g, b], axis=0))
        if set_hdr_props:
            dst.props["_Primaries"] = int(vs.PRIMARIES_BT2020)
            dst.props["_Transfer"] = int(vs.TRANSFER_ST2084)
            dst.props["_Matrix"] = int(vs.MATRIX_RGB)
            dst.props["_ColorRange"] = int(vs.RANGE_FULL)
        return dst

    out = core.std.BlankClip(clip=src, format=rgb30.id)
    return core.std.ModifyFrame(clip=out, clips=[src], selector=_ev_debug(ev))
