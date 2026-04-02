# ⚡ Rekrea Scripts: Desktop Pipelines

The `scripts/` directory provides standalone, GUI-based applications for local media processing. These are the "final form" of the logic explored in the notebooks.

| Script | Functionality |
|---|---|
| `background_removal_pipeline.py` | Background removal with Tkinter GUI |
| `video_enhancement_pipeline.py` | Enhancement with model/scale/tile settings |

Output is written to `scripts/playground_data/output/` (not tracked by git).

## 🐧 Platform Support
These scripts are designed for **Native Linux** environments. For Windows users, we recommend **WSL2 (Windows Subsystem for Linux)** to ensure proper handling of Python-TK and FFmpeg subprocesses.

## 🏗 The Local Environment
The scripts look for a specific directory structure to manage assets. When running a script, ensure your local `rekrea` base folder is organized as follows:

```text
rekrea/
├── models/             # Pre-trained .pth files
├── background_removal/ # Inputs and Outputs for removal tasks
└── enhancement/        # Inputs and Outputs for upscaling tasks
