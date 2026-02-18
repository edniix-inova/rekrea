#!/usr/bin/env bash
# WSL2 system dependencies for Rekrea
# Run once after setting up your WSL2 environment:
#   bash requirements/wsl.sh
#
# Tested on Ubuntu 22.04 (WSL2, kernel 5.15 microsoft-standard-WSL2)

set -e

sudo apt update

# FFmpeg — required for video frame extraction and reassembly.
# ffmpeg-python (pip) is only a wrapper; the actual ffmpeg and ffprobe
# binaries must be present on the system PATH.
sudo apt install -y ffmpeg

# Tkinter — Python's standard GUI toolkit. Not included in some minimal
# Python installations; needed by scripts that use tkinter dialogs.
sudo apt install -y python3-tk

# --- GUI display (required for tkinter on WSL2) ---
#
# Windows 11 and recent Windows 10 builds ship WSLg, which provides a
# built-in Wayland/X11 display server. If 'echo $DISPLAY' prints something
# like ':0' you already have it and no further action is needed.
#
# If $DISPLAY is empty, install an X server on the Windows side:
#   - VcXsrv  (open-source): https://sourceforge.net/projects/vcxsrv/
#   - then add to ~/.bashrc:
#       export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0
#       export LIBGL_ALWAYS_INDIRECT=1
#
# To verify your display is working:
#   sudo apt install -y x11-apps && xeyes

echo ""
echo "System dependencies installed."
echo "Next: pip install -r requirements/pip.txt"
echo ""
echo "To verify display (needed for tkinter):"
echo "  echo \$DISPLAY   # should not be empty"
