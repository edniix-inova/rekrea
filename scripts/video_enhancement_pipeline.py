"""
Video Enhancement Pipeline — Local GUI
=======================================

Selects a video via a file dialog, enhances it using Real-ESRGAN
(frame restoration and upscaling), reattaches the original audio, and
saves the result to scripts/playground_data/output/.

A settings dialog lets the user choose the model, output scale, and tile
size before processing starts.

Usage
-----
    python scripts/video_enhancement_pipeline.py

Requirements
------------
    pip install realesrgan basicsr opencv-python-headless tqdm
    pip install ffmpeg-python
    # torch: install separately with the correct CUDA version for your GPU.
    #   See https://pytorch.org/get-started/locally/
    # ffmpeg system binary: apt install ffmpeg / winget install ffmpeg
"""

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Allow running from the project root or from scripts/ directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rekrea.modules.video_enhancement import MODEL_CONFIGS, process_video

PLAYGROUND_DATA = Path(__file__).parent / "playground_data"
INPUT_DIR = PLAYGROUND_DATA / "input"
OUTPUT_DIR = PLAYGROUND_DATA / "output"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_pipeline(
    input_path: Path,
    output_path: Path,
    model_name: str,
    outscale: float,
    tile: int,
    progress_queue: "queue.Queue[tuple]",
) -> None:
    """Run process_video in a background thread, posting updates to the queue.

    Messages posted to *progress_queue*:
    - ``("status", message: str)`` at each pipeline stage.
    - ``("progress", current: int, total: int)`` after each enhanced frame.
    - ``("done", output_path: str)`` on success.
    - ``("error", message: str)`` on failure.
    """
    def on_progress(current: int, total: int) -> None:
        progress_queue.put(("progress", current, total))

    def on_status(msg: str) -> None:
        progress_queue.put(("status", msg))

    try:
        process_video(
            input_path,
            output_path,
            model_name=model_name,
            outscale=outscale,
            tile=tile,
            reattach_audio=True,
            progress_callback=on_progress,
            status_callback=on_status,
        )
        progress_queue.put(("done", str(output_path)))
    except Exception as exc:
        progress_queue.put(("error", str(exc)))


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

def _settings_dialog(root: tk.Tk) -> tuple | None:
    """Show enhancement settings and return ``(model_name, outscale, tile)``.

    Returns None if the user cancels.
    """
    dialog = tk.Toplevel(root)
    dialog.title("Enhancement settings")
    dialog.resizable(False, False)

    pad = {"padx": 14, "pady": 5}

    # Model selection
    tk.Label(dialog, text="Model:", **pad).grid(row=0, column=0, sticky="w")
    model_var = tk.StringVar(value="RealESRGAN_x4plus")
    model_menu = ttk.Combobox(
        dialog,
        textvariable=model_var,
        values=list(MODEL_CONFIGS),
        state="readonly",
        width=34,
    )
    model_menu.grid(row=0, column=1, **pad)

    # Model description label — updates on selection change
    desc_var = tk.StringVar(value=MODEL_CONFIGS["RealESRGAN_x4plus"]["description"])
    desc_label = tk.Label(
        dialog, textvariable=desc_var, wraplength=320,
        justify="left", fg="#555555", **pad
    )
    desc_label.grid(row=1, column=0, columnspan=2, sticky="w")

    def on_model_change(*_):
        desc_var.set(MODEL_CONFIGS[model_var.get()]["description"])

    model_var.trace_add("write", on_model_change)

    # Output scale
    tk.Label(dialog, text="Output scale:", **pad).grid(row=2, column=0, sticky="w")
    scale_var = tk.DoubleVar(value=2.0)
    ttk.Spinbox(
        dialog, from_=1.0, to=4.0, increment=0.5,
        textvariable=scale_var, width=8,
    ).grid(row=2, column=1, sticky="w", **pad)
    tk.Label(
        dialog,
        text="2× recommended. 4× requires >16 GB VRAM.",
        fg="#555555", **pad
    ).grid(row=3, column=0, columnspan=2, sticky="w")

    # Tile size
    tk.Label(dialog, text="Tile size (0 = off):", **pad).grid(row=4, column=0, sticky="w")
    tile_var = tk.IntVar(value=0)
    ttk.Spinbox(
        dialog, from_=0, to=2048, increment=256,
        textvariable=tile_var, width=8,
    ).grid(row=4, column=1, sticky="w", **pad)
    tk.Label(
        dialog,
        text="Use 512 or 1024 if you run out of VRAM.",
        fg="#555555", **pad
    ).grid(row=5, column=0, columnspan=2, sticky="w")

    # Buttons
    result = {"ok": False}

    def on_ok():
        result["ok"] = True
        dialog.destroy()

    def on_cancel():
        dialog.destroy()

    btn_frame = tk.Frame(dialog)
    btn_frame.grid(row=6, column=0, columnspan=2, pady=10)
    ttk.Button(btn_frame, text="Start", command=on_ok).pack(side="left", padx=8)
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=8)

    dialog.grab_set()
    root.wait_window(dialog)

    if not result["ok"]:
        return None
    return model_var.get(), scale_var.get(), tile_var.get()


# ---------------------------------------------------------------------------
# Progress window
# ---------------------------------------------------------------------------

def _build_progress_window(root: tk.Tk, filename: str) -> tuple:
    """Create and return a progress window.

    Returns
    -------
    (window, progress_var, status_var)
    """
    win = tk.Toplevel(root)
    win.title("Rekrea — Video Enhancement")
    win.resizable(False, False)
    # Prevent closing while processing
    win.protocol("WM_DELETE_WINDOW", lambda: None)

    padding = {"padx": 20, "pady": 6}

    tk.Label(win, text=f"Processing: {filename}", font=("", 10, "bold"), **padding).pack()

    progress_var = tk.DoubleVar(value=0)
    bar = ttk.Progressbar(win, variable=progress_var, maximum=100, length=400)
    bar.pack(padx=20, pady=4)

    status_var = tk.StringVar(value="Starting…")
    tk.Label(win, textvariable=status_var, **padding).pack()

    win.update()
    return win, progress_var, status_var


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.withdraw()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Video selection ---
    raw_path = filedialog.askopenfilename(
        title="Select a video to enhance",
        initialdir=str(INPUT_DIR),
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ],
    )
    if not raw_path:
        root.destroy()
        return

    input_path = Path(raw_path)

    # --- Settings ---
    settings = _settings_dialog(root)
    if settings is None:
        root.destroy()
        return

    model_name, outscale, tile = settings
    output_path = OUTPUT_DIR / f"{input_path.stem}_enhanced_x{outscale:.0f}.mp4"

    # --- Progress window ---
    win, progress_var, status_var = _build_progress_window(root, input_path.name)

    # --- Launch background thread ---
    progress_queue: "queue.Queue[tuple]" = queue.Queue()
    thread = threading.Thread(
        target=_run_pipeline,
        args=(input_path, output_path, model_name, outscale, tile, progress_queue),
        daemon=True,
    )
    thread.start()

    # --- Poll queue and update UI ---
    def poll() -> None:
        try:
            while True:
                msg = progress_queue.get_nowait()

                if msg[0] == "status":
                    status_var.set(msg[1])

                elif msg[0] == "progress":
                    _, current, total = msg
                    pct = (current / total) * 100
                    progress_var.set(pct)
                    status_var.set(f"Enhancing frame {current} of {total}  ({pct:.0f}%)")

                elif msg[0] == "done":
                    win.protocol("WM_DELETE_WINDOW", win.destroy)
                    messagebox.showinfo(
                        "Done",
                        f"Output saved to:\n{msg[1]}",
                        parent=win,
                    )
                    win.destroy()
                    root.destroy()
                    return

                elif msg[0] == "error":
                    win.protocol("WM_DELETE_WINDOW", win.destroy)
                    messagebox.showerror("Processing failed", msg[1], parent=win)
                    win.destroy()
                    root.destroy()
                    return

        except queue.Empty:
            pass

        root.after(100, poll)

    root.after(100, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
