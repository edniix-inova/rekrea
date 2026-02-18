"""
Background Removal Pipeline — Local GUI
========================================

Selects a video via a file dialog, removes the background from each frame
using the rekrea background_removal module, and saves the result to
scripts/playground_data/output/.

Usage
-----
    python scripts/background_removal_pipeline.py

Requirements
------------
    pip install rembg ffmpeg-python onnxruntime tqdm
    # ffmpeg must be installed on the system (apt install ffmpeg / winget install ffmpeg)
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

from rekrea.modules.background_removal import process_video

PLAYGROUND_DATA = Path(__file__).parent / "playground_data"
INPUT_DIR = PLAYGROUND_DATA / "input"
OUTPUT_DIR = PLAYGROUND_DATA / "output"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_pipeline(
    input_path: Path,
    output_path: Path,
    progress_queue: "queue.Queue[tuple]",
) -> None:
    """Run process_video in a background thread, posting updates to the queue.

    Messages posted to *progress_queue*:
    - ``("progress", current: int, total: int)`` after each frame.
    - ``("done", output_path: str)`` on success.
    - ``("error", message: str)`` on failure.
    """
    def on_progress(current: int, total: int) -> None:
        progress_queue.put(("progress", current, total))

    try:
        process_video(input_path, output_path, progress_callback=on_progress)
        progress_queue.put(("done", str(output_path)))
    except Exception as exc:
        progress_queue.put(("error", str(exc)))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def _build_progress_window(root: tk.Tk, filename: str) -> tuple:
    """Create and return a progress window with a progress bar and status label.

    Returns
    -------
    (window, progress_var, status_var)
    """
    win = tk.Toplevel(root)
    win.title("Rekrea — Background Removal")
    win.resizable(False, False)
    # Prevent the user from closing the window while processing
    win.protocol("WM_DELETE_WINDOW", lambda: None)

    padding = {"padx": 20, "pady": 6}

    tk.Label(win, text=f"Processing: {filename}", font=("", 10, "bold"), **padding).pack()

    progress_var = tk.DoubleVar(value=0)
    bar = ttk.Progressbar(win, variable=progress_var, maximum=100, length=400)
    bar.pack(padx=20, pady=4)

    status_var = tk.StringVar(value="Extracting frames…")
    tk.Label(win, textvariable=status_var, **padding).pack()

    win.update()
    return win, progress_var, status_var


def main() -> None:
    root = tk.Tk()
    root.withdraw()  # Hide the root window; we only use dialogs and Toplevel

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Video selection ---
    raw_path = filedialog.askopenfilename(
        title="Select a video to process",
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
    output_path = OUTPUT_DIR / f"{input_path.stem}_no_bg.mp4"

    # --- Progress window ---
    win, progress_var, status_var = _build_progress_window(root, input_path.name)

    # --- Launch background thread ---
    progress_queue: "queue.Queue[tuple]" = queue.Queue()
    thread = threading.Thread(
        target=_run_pipeline,
        args=(input_path, output_path, progress_queue),
        daemon=True,
    )
    thread.start()

    # --- Poll queue and update UI ---
    def poll() -> None:
        try:
            while True:
                msg = progress_queue.get_nowait()

                if msg[0] == "progress":
                    _, current, total = msg
                    pct = (current / total) * 100
                    progress_var.set(pct)
                    status_var.set(f"Frame {current} of {total}  ({pct:.0f}%)")

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
