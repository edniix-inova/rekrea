# 📓 Rekrea Notebooks: The Learning Lab

These notebooks serve as the educational entry point for Rekrea. Each one is a self-contained "lesson" that combines theory, code, and visual results.

## ☁️ The Colab Environment
To ensure persistence and ease of use, these notebooks are designed to interface with a **Rekrea Environment** on Google Drive.

**Setup Instructions:**
1.  Mount your Google Drive within the notebook.
2.  The notebooks expect a base directory at `MyDrive/rekrea/`.
3.  This structure allows you to keep your `models/` (weights) and `input/output` data synced across sessions.

## 🧬 Available Modules
| Notebook | Concept | Core Model |
|---|---|---|
| `VideoEditing_BackgroundRemoval_RemBG_Pipeline.ipynb` | Neural Segmentation | rembg / U2Net |
| `VideoEditing_Enhancement_RealESRGAN_Pipeline.ipynb` | Super-Resolution | Real-ESRGAN |
| `VideoEditing_Interpolation_VFI_Pipelne.ipynb` | Temporal Consistency | VFIformer |

*Auxiliary scripts used for Colab-specific batching reside in the `auxiliary/` folder.*
