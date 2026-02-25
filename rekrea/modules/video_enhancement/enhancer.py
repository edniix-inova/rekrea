"""
Video Enhancement Module (Real-ESRGAN)
=======================================

Restores and upscales video frames using Real-ESRGAN, a GAN trained with
a realistic degradation model to reverse compression artefacts, blur, and
noise from real-world footage.

How it works
------------
Each frame is passed through a pre-trained RRDBNet generator that produces
an upscaled output. The model operates at 4× internally and can be
downsampled to a chosen ``outscale`` (e.g. 2×) to stay within VRAM limits.

Unlike background removal, Real-ESRGAN uses a *blind* restoration approach:
it does not know what degradations affected the input and instead applies a
combination of super-resolution, deblurring, and denoising learned from
a synthetic degradation pipeline during training.

Video I/O (frame extraction, reassembly, audio muxing) is handled by the
shared utilities in rekrea.utils.video.

Installation
------------
This module requires the ``realesrgan`` pip package and PyTorch::

    pip install realesrgan basicsr
    # Install torch separately with the correct CUDA version:
    # https://pytorch.org/get-started/locally/

References
----------
Real-ESRGAN paper: "Real-ESRGAN: Training Real-World Blind Super-Resolution
with Pure Synthetic Data", Wang et al., 2021.
https://arxiv.org/abs/2107.10833

Repository: https://github.com/xinntao/Real-ESRGAN
"""

from pathlib import Path
from typing import Callable, Optional
import shutil
import sys
import tempfile
import types

# ---------------------------------------------------------------------------
# Compatibility shim — must run before any basicsr/realesrgan import.
#
# basicsr.data.degradations has a hardcoded:
#   from torchvision.transforms.functional_tensor import rgb_to_grayscale
# That private submodule was removed in torchvision >= 0.17.0 and the
# function moved to torchvision.transforms.v2.functional.
#
# We install a thin shim module under the old name so basicsr finds it,
# without patching any files on disk (unlike the sed approach used in Colab).
# ---------------------------------------------------------------------------
if "torchvision.transforms.functional_tensor" not in sys.modules:
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        from torchvision.transforms.v2 import functional as _tvf
        _shim = types.ModuleType("torchvision.transforms.functional_tensor")
        _shim.rgb_to_grayscale = _tvf.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = _shim
        del _shim, _tvf

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from tqdm import tqdm

from rekrea.utils.video import extract_frames, mux_audio, rebuild_video


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "description": "General-purpose restoration and 4× upscaling. Recommended default.",
    },
    "RealESRNet_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "description": "Smoother output with fewer GAN artefacts. Use when x4plus over-sharpens.",
    },
    "RealESRGAN_x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
        "description": "2× upscaling. Faster and lower VRAM than the 4× variants.",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
        "description": "Optimised for anime and illustration content (lighter: 6 RRDB blocks).",
    },
}

DEFAULT_MODEL = "RealESRGAN_x4plus"


# ---------------------------------------------------------------------------
# Public pipeline steps
# ---------------------------------------------------------------------------

def create_upsampler(
    model_name: str = DEFAULT_MODEL,
    tile: int = 0,
) -> RealESRGANer:
    """Initialise and return a :class:`RealESRGANer` upsampler.

    On first use the model weights are downloaded automatically and cached
    by ``basicsr`` (typically in ``~/.cache/basicsr/``). GPU (CUDA) is used
    automatically if available; falls back to CPU otherwise.

    Parameters
    ----------
    model_name:
        Key from :data:`MODEL_CONFIGS`. Default ``"RealESRGAN_x4plus"``.
    tile:
        Tile size for tiled inference. ``0`` = process the full frame at
        once (fastest, most VRAM). Use ``512`` or ``1024`` if full-frame
        inference runs out of memory.

    Returns
    -------
    RealESRGANer
        Ready to call ``.enhance()`` on individual frames.

    Raises
    ------
    ValueError
        If *model_name* is not found in :data:`MODEL_CONFIGS`.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_CONFIGS)}"
        )
    cfg = MODEL_CONFIGS[model_name]
    # FP16 (half precision) halves VRAM usage but is only supported on GPU
    half = torch.cuda.is_available()

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=cfg["num_block"],
        num_grow_ch=32,
        scale=cfg["scale"],
    )
    return RealESRGANer(
        scale=cfg["scale"],
        model_path=cfg["url"],
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
    )


def enhance_frames(
    frames_dir: Path,
    output_dir: Path,
    upsampler: RealESRGANer,
    outscale: float = 2.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Apply Real-ESRGAN to every PNG frame in *frames_dir*.

    The upsampler is provided pre-initialised (from :func:`create_upsampler`)
    so that the model is loaded once and reused across all frames.

    Output frames keep the same filename as the input frames, so the
    ``frame_XXXXX.png`` naming convention is preserved for later reassembly.

    Parameters
    ----------
    frames_dir:
        Directory containing input frames (``frame_XXXXX.png``).
    output_dir:
        Directory for enhanced frames. Created if needed.
    upsampler:
        Initialised :class:`RealESRGANer` instance.
    outscale:
        Final output scale factor. The model processes at its internal
        scale (4× for x4plus) then downsamples to this value. Recommend
        ``2.0`` for most GPUs; ``4.0`` requires >16 GB VRAM.
    progress_callback:
        Optional function called with ``(current, total)`` after each frame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    total = len(frame_files)

    for i, frame_path in enumerate(tqdm(frame_files, desc="Enhancing frames"), start=1):
        # cv2 reads PNG as BGR — RealESRGANer expects BGR input
        img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        output, _ = upsampler.enhance(img, outscale=outscale)
        cv2.imwrite(str(output_dir / frame_path.name), output)

        if progress_callback:
            progress_callback(i, total)


def process_video(
    input_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    outscale: float = 2.0,
    tile: int = 0,
    reattach_audio: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Run the full Real-ESRGAN enhancement pipeline on a video file.

    Steps performed:
    1. Extract frames from *input_path*.
    2. Initialise the Real-ESRGAN model (downloads weights on first use).
    3. Enhance each frame (restoration + upscaling).
    4. Reassemble enhanced frames into a silent MP4.
    5. Optionally mux the original audio back onto the output.

    All temporary data is stored in a system temp directory and removed
    automatically when processing completes or if an error occurs.

    Parameters
    ----------
    input_path:
        Path to the source video.
    output_path:
        Path for the resulting video. Parent directories are created if needed.
    model_name:
        Real-ESRGAN model key from :data:`MODEL_CONFIGS`.
    outscale:
        Output scale factor. ``2.0`` recommended for most GPUs.
    tile:
        Tile size for tiled inference. ``0`` = full-frame (default).
        Use ``512`` or ``1024`` if you run out of VRAM.
    reattach_audio:
        If True, mux the original audio from *input_path* back onto the
        output. Default True.
    progress_callback:
        Called with ``(current_frame, total_frames)`` after each enhanced frame.
    status_callback:
        Called with a human-readable status string at each pipeline stage.
        Useful for updating a GUI label (e.g. "Loading model…").
    """
    def _notify(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    with tempfile.TemporaryDirectory(prefix="rekrea_esrgan_") as tmp:
        tmp_path = Path(tmp)
        frames_dir = tmp_path / "frames"
        enhanced_dir = tmp_path / "enhanced_frames"
        silent_output = tmp_path / "silent_output.mp4"

        _notify("Extracting frames…")
        framerate = extract_frames(input_path, frames_dir)

        _notify("Loading Real-ESRGAN model…")
        upsampler = create_upsampler(model_name, tile=tile)

        _notify("Enhancing frames…")
        enhance_frames(frames_dir, enhanced_dir, upsampler, outscale, progress_callback)

        _notify("Rebuilding video…")
        rebuild_video(enhanced_dir, silent_output, framerate)

        if reattach_audio:
            _notify("Reattaching audio…")
            mux_audio(silent_output, input_path, output_path)
        else:
            shutil.copy2(str(silent_output), str(output_path))
