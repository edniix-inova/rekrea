# Rekrea

**Rekrea** is a modular video editing toolkit that integrates AI-powered techniques into media creation workflows. It provides reusable Python modules, ready-to-run pipeline scripts, and didactic Colab notebooks covering background removal, video enhancement, and frame interpolation.

---

## Repository Structure

```
rekrea/
├── notebooks/              # Didactic Colab notebooks + auxiliary scripts
│   └── auxiliary/          # Wrapper scripts used by notebooks (e.g. Colab helpers)
├── rekrea/                 # Python package — modules and shared utilities
│   ├── modules/
│   │   ├── background_removal/
│   │   └── video_enhancement/
│   └── utils/
├── scripts/                # Standalone pipeline scripts with GUI
└── requirements/           # Dependency files
```

---

## Functionalities

### Background Removal
Removes backgrounds from every frame of a video using neural segmentation. Powered by [rembg](https://github.com/danielgatis/rembg) with U2Net and BiRefNet models. Outputs video frames with transparency preserved as PNG before final reassembly.

### Video Enhancement
Upscales and restores video quality using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Supports 2× and 4× upscaling with models tuned for general footage or anime/illustration content. Handles large frames via tiled inference to stay within VRAM limits.

### Frame Interpolation
Generates intermediate frames between existing ones to increase effective frame rate. Uses [VFIformer](https://github.com/dvlab-research/VFIformer) for high-quality temporal interpolation. Demonstrated in the Colab notebook with a batch wrapper for efficient processing.

---

## Notebooks

Interactive Colab notebooks under `notebooks/` explain each functionality step by step, covering model setup, parameter choices, and expected results. They are intended as both learning material and starting points for custom workflows.

| Notebook | Functionality | Model |
|---|---|---|
| `VideoEditing_BackgroundRemoval_RemBG_Pipeline.ipynb` | Background removal | rembg / U2Net |
| `VideoEditing_Enhancement_RealESRGAN_Pipeline.ipynb` | Video enhancement | Real-ESRGAN |
| `VideoEditing_Interpolation_VFI_Pipelne.ipynb` | Frame interpolation | VFIformer |

Auxiliary scripts used by the notebooks (e.g. `vfiformer_batch_wrapper.py`) live in `notebooks/auxiliary/`.

---

## Modules

The `rekrea/` package exposes each functionality as an independent module. All modules share common video I/O utilities backed by FFmpeg.

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

---

## Scripts

`scripts/` contains GUI pipeline applications for local use. Run them directly — a file-picker dialog opens, then a settings dialog (where applicable), followed by a live progress window.

| Script | Functionality |
|---|---|
| `background_removal_pipeline.py` | Background removal with Tkinter GUI |
| `video_enhancement_pipeline.py` | Enhancement with model/scale/tile settings |

Output is written to `scripts/playground_data/output/` (not tracked by git).

---

## Environment Layout

Both notebooks and scripts expect a **rekrea base directory** (either `MyDrive/rekrea/` in Colab or a local path) with the following layout:

```
rekrea/
├── models/                         # Pre-trained model weights
├── scripts/                        # Auxiliary scripts (e.g. Colab wrappers)
└── <functionality>/                # One folder per functionality
    └── <method>/                   # One folder per method/model variant
        ├── input/                  # Source material
        └── output/                 # Processed results
```

**Example:**

```
rekrea/
├── models/
│   └── RealESRGAN_x4plus.pth
├── scripts/
│   └── vfiformer_batch_wrapper.py
├── background_removal/
│   └── rembg/
│       ├── input/
│       └── output/
├── interpolation/
│   └── vfiformer/
│       ├── input/
│       └── output/
└── enhancement/
    └── realesrgan/
        ├── input/
        └── output/
```

---

## Installation

### System dependencies (Linux / WSL2)

```bash
bash requirements/wsl.sh
```

This installs `ffmpeg` and `python3-tk`.

### Python dependencies

Install PyTorch first using the [official selector](https://pytorch.org/get-started/locally/) for your platform and CUDA version, then:

```bash
pip install -r requirements/pip.txt
```

For GPU inference, replace `onnxruntime` with `onnxruntime-gpu` in `requirements/pip.txt` before installing.

---

## License

This project is for educational and personal use. Each AI model integrated here carries its own license — refer to the respective upstream repositories for terms.
