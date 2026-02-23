"""
Video Utilities
===============

General-purpose video I/O and frame manipulation utilities shared across
all Rekrea modules. All video processing delegates to FFmpeg via the
ffmpeg-python library.

Used by
-------
- rekrea.modules.background_removal
- rekrea.modules.video_enhancement

Functions
---------
get_video_info    — probe video metadata including VFR/CFR detection
extract_frames    — decode a video into individual PNG frames
rebuild_video     — reassemble PNG frames into an H.264 MP4
mux_audio         — attach an audio track from a source video to a silent video
downscale_frames  — batch-downscale frames using the Lanczos filter
"""

from pathlib import Path
from typing import Optional

import ffmpeg


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_video_info(video_path: Path) -> dict:
    """Probe *video_path* and return a dict of video metadata.

    Compares the declared frame rate (``r_frame_rate``) with the average
    frame rate (``avg_frame_rate``) to detect Variable Frame Rate (VFR)
    content. A discrepancy greater than 1 % flags VFR.

    VFR is common in phone recordings and screen captures. Reassembling a
    VFR video with a fixed frame rate produces incorrect timing — frames
    play too fast or too slow. Callers should inspect ``vfr_suspected``
    before proceeding to frame reassembly.

    Parameters
    ----------
    video_path:
        Path to the video file to probe.

    Returns
    -------
    dict with keys:

    * ``framerate`` – declared real frame rate (``r_frame_rate``), float.
    * ``avg_framerate`` – average frame rate (``avg_frame_rate``), float.
    * ``vfr_suspected`` – True if the two rates differ by more than 1 %.
    * ``cfr`` – the confirmed constant frame rate, or None if VFR suspected.
    """
    probe = ffmpeg.probe(str(video_path))
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

    def _parse(rate_str: str) -> float:
        num, den = rate_str.split("/")
        return int(num) / max(int(den), 1)

    framerate = _parse(stream["r_frame_rate"])
    avg_framerate = _parse(stream.get("avg_frame_rate") or stream["r_frame_rate"])
    vfr_suspected = abs(framerate - avg_framerate) / max(framerate, 1) > 0.01

    return {
        "framerate": framerate,
        "avg_framerate": avg_framerate,
        "vfr_suspected": vfr_suspected,
        "cfr": None if vfr_suspected else framerate,
    }


# ---------------------------------------------------------------------------
# Frame extraction and reassembly
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path) -> float:
    """Decode *video_path* into individual PNG frames saved in *frames_dir*.

    Frames are named ``frame_00001.png``, ``frame_00002.png``, … to
    preserve display order. The ``-vsync 0`` flag ensures one output file
    per decoded frame, which matters for variable-fps containers.

    Parameters
    ----------
    video_path:
        Source video file.
    frames_dir:
        Output directory. Created if it does not exist.

    Returns
    -------
    float
        The video's declared framerate. Pass this to :func:`rebuild_video`
        so the reassembled video plays at the correct speed.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    info = get_video_info(video_path)
    (
        ffmpeg
        .input(str(video_path))
        .output(str(frames_dir / "frame_%05d.png"), pix_fmt="rgb24", qscale=1)
        .global_args("-vsync", "0")
        .overwrite_output()
        .run(quiet=True)
    )
    return info["framerate"]


def rebuild_video(
    frames_dir: Path,
    output_path: Path,
    framerate: float,
    crf: int = 16,
    preset: str = "slow",
) -> None:
    """Reassemble PNG frames from *frames_dir* into an H.264 MP4.

    ``yuv420p`` is required for H.264 compatibility. It does not carry an
    alpha channel, so any RGBA transparency in the source frames is
    composited over black in the output. Use VP9 + WebM for transparent
    output.

    ``movflags=+faststart`` moves the container index to the start of the
    file so the video can begin playing while it is still downloading.

    Parameters
    ----------
    frames_dir:
        Directory containing ``frame_XXXXX.png`` files.
    output_path:
        Destination MP4 path. Parent directories are created if needed.
    framerate:
        Frames per second — must match the value returned by
        :func:`extract_frames` so playback speed is correct.
    crf:
        H.264 Constant Rate Factor. 0 = lossless, 51 = worst quality.
        16 gives near-lossless results at a manageable file size.
    preset:
        FFmpeg encoding preset. Slower presets produce smaller files at the
        same quality level without changing the CRF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(frames_dir / "frame_%05d.png"), framerate=framerate)
        .output(
            str(output_path),
            vcodec="libx264",
            crf=crf,
            preset=preset,
            pix_fmt="yuv420p",
            movflags="+faststart",
        )
        .overwrite_output()
        .run(quiet=True)
    )


# ---------------------------------------------------------------------------
# Audio muxing
# ---------------------------------------------------------------------------

def mux_audio(
    video_path: Path,
    audio_source: Path,
    output_path: Path,
) -> None:
    """Attach the audio stream from *audio_source* to *video_path*.

    The video stream is stream-copied (no re-encoding, no quality loss).
    ``-shortest`` trims the output to the shorter of the two streams,
    handling minor duration drift that arises when frame counts change
    during processing (e.g. after interpolation).

    This is equivalent to:
    ``ffmpeg -i video.mp4 -i audio_source.mp4 -map 0:v:0 -map 1:a:0
    -c:v copy -shortest output.mp4``

    Parameters
    ----------
    video_path:
        The processed (silent) video whose video stream will be used.
    audio_source:
        The original video whose audio track will be reused.
    output_path:
        Destination for the muxed output. Parent dirs created if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    in_video = ffmpeg.input(str(video_path))
    in_audio = ffmpeg.input(str(audio_source))
    (
        ffmpeg
        .output(
            in_video.video,
            in_audio.audio,
            str(output_path),
            vcodec="copy",
            acodec="copy",
            shortest=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def downscale_frames(
    frames_dir: Path,
    output_dir: Path,
    target_height: int,
    target_width: int = -2,
) -> None:
    """Batch-downscale PNG frames using the Lanczos filter.

    ``target_width=-2`` (default) tells FFmpeg to compute the width that
    preserves the aspect ratio while rounding to the nearest even number —
    required for H.264 compatibility.

    Useful for reducing resolution before frame interpolation (e.g. RIFE)
    so that inference stays within VRAM limits, with a second upscaling
    pass afterwards to recover resolution.

    Parameters
    ----------
    frames_dir:
        Source directory containing ``frame_XXXXX.png`` files.
    output_dir:
        Destination directory. Created if needed.
    target_height:
        Target height in pixels (e.g. 1920 for portrait HD).
    target_width:
        Target width. -2 = preserve aspect ratio (default).
    """
    if not any(frames_dir.glob("frame_*.png")):
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(frames_dir / "frame_%05d.png"))
        .filter("scale", target_width, target_height, flags="lanczos")
        .output(str(output_dir / "frame_%05d.png"))
        .overwrite_output()
        .run(quiet=True)
    )
