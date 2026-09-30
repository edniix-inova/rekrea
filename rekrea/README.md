# 📦 Rekrea Python Modules

This directory contains the `rekrea` core package—a set of modular, reusable Python components for media processing. These modules strip away the "notebook" explanations to provide clean, programmatic access to AI functions.

## 🛠 Core Components

### `rekrea.modules`
* **`background_removal`**: High-level wrappers for U2Net and BiRefNet.
* **`video_enhancement`**: Tiled inference for upscaling (Real-ESRGAN).

### `rekrea.utils`
A robust set of **FFmpeg-backed** utilities for video I/O, ensuring that transparency (alpha channels) and audio are preserved during the AI transformation process.

## 🔎 Core Components in Detail

### `rekrea.modules.background_removal`

```python
from rekrea.modules.background_removal import process_video

process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    model="u2net",           # u2net | u2netp | isnet-general-use | birefnet-general
    progress_callback=None,
)
```

Key functions: `process_video`, `remove_background_from_frames`, `extract_frames`, `rebuild_video`.

### `rekrea.modules.video_enhancement`

```python
from rekrea.modules.video_enhancement import process_video, MODEL_CONFIGS

process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    model_name="RealESRGAN_x4plus",   # see MODEL_CONFIGS for all options
    outscale=4,
    tile=512,                          # reduce if VRAM is limited; 0 = no tiling
    progress_callback=None,
)
```

Available models:

| Model | Scale | Notes |
|---|---|---|
| `RealESRGAN_x4plus` | 4× | General purpose (default) |
| `RealESRNet_x4plus` | 4× | Fewer GAN artifacts |
| `RealESRGAN_x2plus` | 2× | Faster, lower memory |
| `RealESRGAN_x4plus_anime_6B` | 4× | Anime / illustration |

Key functions: `process_video`, `enhance_frames`, `create_upsampler`.

### `rekrea.utils.video`

Shared FFmpeg utilities used internally by all modules:

| Function | Description |
|---|---|
| `extract_frames` | Decode video to PNG frames |
| `rebuild_video` | Reassemble frames to H.264 MP4 |
| `mux_audio` | Attach audio track from source video |
| `get_video_info` | Probe metadata, detect VFR/CFR |
| `downscale_frames` | Batch downscale frames (Lanczos) |
