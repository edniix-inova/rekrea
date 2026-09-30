# Rekrea: Applied AI for Creative Systems Literacy

**Rekrea** is an interactive platform designed to bridge the gap between "black-box" AI tools and transparent, reproducible creative workflows. Part of the [Edniix Inova](https://github.com/edniix-inova) ecosystem, it empowers artists, hobbists, and young future developers to build technical intuition through hands-on media manipulation.

## 🌟 The Vision
In an era of generative AI, understanding the *mechanics* of the pipeline is as important as the output. Rekrea is structured as a didactic journey:
1.  **Learn** the theory and logic via interactive notebooks.
2.  **Integrate** logic into custom systems using modular Python components.
3.  **Execute** professional-grade pipelines through standalone scripts.

---

## 🏛 Project Architecture: The Three Pillars

Rekrea is organized into three distinct layers, moving from educational exploration to production-ready execution:

### 1. [The Notebooks](./notebooks/) (Learn)
Didactic Google Colab environments that break down complex AI functions (Background Removal, Enhancement, Interpolation) into step-by-step logic.
* **Best for:** Learning, experimentation, and cloud-based prototyping.

### 2. [The Modules](./rekrea/) (Build)
The core `rekrea` Python package. This is a modular implementation of the functions explored in the notebooks, designed to be imported into larger pipelines.
* **Best for:** Developers building custom AI-driven software.

### 3. [The Scripts](./scripts/) (Execute)
Ready-to-run pipeline implementations with GUI interfaces. These leverage the modules to provide a local, high-performance desktop experience.
* **Best for:** Batch processing and local production workflows.

---

## 🗂️ Environment Layout

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

## 🪪 License

This project is for educational and personal use. Each AI model integrated here carries its own license — refer to the respective upstream repositories for terms.
