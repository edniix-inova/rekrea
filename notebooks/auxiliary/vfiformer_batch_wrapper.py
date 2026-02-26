#!/usr/bin/env python3
"""
VFIformer Colab Batch Wrapper

Interpolate between all frames in a directory using VFIformer and demo.py.
Saves intermediate frames between each pair without renaming originals.

demo.py: https://github.com/JIA-Lab-research/VFIformer/ 
net_220.pth: https://huggingface.co/xmanifold/vfiformer/tree/main/pretrained_VFIformer
"""

import subprocess
import cv2
from pathlib import Path

# Paths (customize if needed)
project_root = Path("/content")
input_dir = project_root / "downscaled_frames"
interp_dir = project_root / "interpolated_frames"
model_path = project_root / "trained_models"
resized_dir = project_root / "resized_frames"

# Create temporary folders for resized frames
if not resized_dir.exists():
    resized_dir.mkdir(parents=True)

# Ensure VFI output folder exists
if not interp_dir.exists():
    interp_dir.mkdir(parents=True)

# Resize frames before feeding into VFIformer
target_resolution = (720,1280)

input_frames = sorted([f for f in input_dir.iterdir() if f.suffix == ".png"])
for f in input_frames:
    img = cv2.imread(str(f))
    resized = cv2.resize(img, target_resolution, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(resized_dir / f.name), resized)

# List input frames
resized_frames = sorted([f for f in resized_dir.iterdir() if f.suffix == ".png"])

print(f"Found {len(resized_frames)} resized frames.")
frame_pairs = zip(resized_frames[:-1], resized_frames[1:])

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