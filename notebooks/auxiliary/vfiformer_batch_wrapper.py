#!/usr/bin/env python3
"""
VFIformer Colab Batch Wrapper

Interpolate between all frames in a directory using VFIformer and demo.py.
Saves intermediate frames between each pair without renaming originals.

Frames are expected to arrive pre-scaled to VFIformer-compatible dimensions
(both width and height divisible by 32). The Real-ESRGAN notebook's Step 5.2a
downscales to 704×1280 via FFmpeg, which satisfies this constraint.

demo.py: https://github.com/JIA-Lab-research/VFIformer/
net_220.pth: https://huggingface.co/xmanifold/vfiformer/tree/main/pretrained_VFIformer
"""

import subprocess
from pathlib import Path

# Paths (customize if needed)
model_name = "net_220.pth"
project_root = Path("/content")
input_dir = project_root / "downscaled_frames"
interp_dir = project_root / "interpolated_frames"
model_path = project_root / "VFIformer/pretrained_models" / model_name

# Ensure VFI output folder exists
if not interp_dir.exists():
    interp_dir.mkdir(parents=True)

# List input frames
input_frames = sorted([f for f in input_dir.iterdir() if f.suffix == ".png"])

print(f"Found {len(input_frames)} input frames.")
frame_pairs = zip(input_frames[:-1], input_frames[1:])

for i, (f1, f2) in enumerate(frame_pairs):
    interp_name = f"interp_{i:05d}.png"
    out_path = interp_dir / interp_name

    demo_script = "/content/VFIformer/demo.py"

    cmd = [
        "python", demo_script,
        "--img0_path", str(f1),
        "--img1_path", str(f2),
        "--resume", str(model_path),
        "--save_folder", str(out_path)
    ]

    print(f"▶️ Interpolating between: {f1.name} and {f2.name}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)

print("✅ All pairs processed.")
