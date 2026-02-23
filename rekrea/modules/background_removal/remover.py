"""
Background Removal Module
=========================

Removes backgrounds from video files using rembg, a library built on neural
segmentation models (U2Net by default).

How it works
------------
Each video frame is decoded into a PNG image and passed through a neural network
(U2Net or an alternative such as BiRefNet) that predicts a per-pixel alpha mask.
White pixels in the mask indicate foreground; black pixels indicate background.
The mask is applied to the original frame as an alpha channel, producing a
transparent PNG. Frames are then re-encoded into a video with FFmpeg.

This is a *naive* (frame-independent) approach — the model processes each frame
with no knowledge of neighbouring frames, which can cause flickering at the
subject boundary.

Video I/O is handled by the shared utilities in rekrea.utils.video.

References
----------
U2Net paper: "U2-Net: Going Deeper with Nested U-Structure for Salient Object
Detection", Qin et al., 2020. https://arxiv.org/abs/2005.09007

rembg library: https://github.com/danielgatis/rembg
"""

from pathlib import Path
from typing import Callable, Optional
import tempfile

from rembg import new_session, remove
from tqdm import tqdm

from rekrea.utils.video import extract_frames, rebuild_video


def remove_background_from_frames(
    frames_dir: Path,
    output_dir: Path,
    model_name: str = "u2net",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Apply background removal to every PNG frame in *frames_dir*.

    The rembg session (model) is created once and reused across all frames,
    avoiding repeated model loading overhead.

    Parameters
    ----------
    frames_dir:
        Directory containing input frames (``frame_XXXXX.png``).
    output_dir:
        Directory where processed frames are written. Created if needed.
    model_name:
        rembg model identifier. Options include:

        - ``"u2net"`` (default) — general-purpose, ~176 MB.
        - ``"u2netp"`` — lighter and faster, lower quality.
        - ``"isnet-general-use"`` — improved segmentation for complex edges.
        - ``"birefnet-general"`` — high-quality, slower, larger model.
    progress_callback:
        Optional function called after each frame with ``(current, total)``
        integers. Useful for progress bars in calling scripts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    total = len(frame_files)

    # Load the model once — avoids repeated disk reads across frames
    session = new_session(model_name)

    for i, frame_path in enumerate(tqdm(frame_files, desc="Removing backgrounds"), start=1):
        input_bytes = frame_path.read_bytes()
        output_bytes = remove(input_bytes, session=session)
        (output_dir / frame_path.name).write_bytes(output_bytes)

        if progress_callback:
            progress_callback(i, total)


def process_video(
    input_path: Path,
    output_path: Path,
    model_name: str = "u2net",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Run the full background-removal pipeline on a video file.

    Steps performed:
    1. Extract frames from *input_path* (via rekrea.utils.video).
    2. Apply background removal to each frame.
    3. Reassemble the processed frames into *output_path*
       (via rekrea.utils.video).

    Temporary frame data is stored in a system temp directory and deleted
    automatically when processing finishes (or if an error occurs).

    Parameters
    ----------
    input_path:
        Path to the source video.
    output_path:
        Path for the resulting video. Parent directories are created if needed.
    model_name:
        rembg model to use (see :func:`remove_background_from_frames`).
    progress_callback:
        Optional progress reporter called with ``(current_frame, total_frames)``
        after each frame is processed.
    """
    with tempfile.TemporaryDirectory(prefix="rekrea_rembg_") as tmp:
        tmp_path = Path(tmp)
        frames_dir = tmp_path / "frames"
        output_frames_dir = tmp_path / "output_frames"

        framerate = extract_frames(input_path, frames_dir)
        remove_background_from_frames(
            frames_dir, output_frames_dir, model_name, progress_callback
        )
        rebuild_video(output_frames_dir, output_path, framerate)
