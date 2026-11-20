# Hand Gesture Recognition

A webcam hand-tracking and gesture-recognition project using OpenCV and
MediaPipe Tasks. It can collect labeled hand-landmark samples, recognize
gestures with a k-nearest-neighbors classifier, and add corrections to the
training data.

## Requirements

- Python 3.10 or later
- A working webcam
- Linux with GUI support for the OpenCV preview window; WSL2 with WSLg is
  supported when the webcam is attached to WSL

Install Python dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Ubuntu/WSL, the OpenCV and MediaPipe native libraries may also require:

```bash
sudo apt update
sudo apt install libsm6 libice6 libgles2
```

## Model

Download the MediaPipe Hand Landmarker model into this folder:

```bash
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o hand_landmarker.task
```

The model is ignored by Git and is not included in the repository.

## Webcam on WSL

Attach the webcam to WSL from Windows PowerShell with `usbipd`:

```powershell
usbipd list
usbipd attach --wsl --busid <BUSID>
```

If the device has not been shared before, run `usbipd bind --busid <BUSID>`
from Administrator PowerShell first. Reattach it after shutting down WSL.

## Platform support

The script currently targets Linux/WSL: it opens the webcam with OpenCV's
V4L2 backend. **Running it natively on Windows is not supported yet** because
that camera backend is Linux-specific. To use it on a Windows computer, run it
inside WSL2, attach the webcam to WSL, and use WSLg for the preview window.

Mouse mode in WSL starts `windows_mouse_bridge.ps1` to move and click the
Windows desktop cursor and detect Ctrl presses. It requires WSL Windows
interop and `powershell.exe` on the WSL `PATH`. On a regular Linux desktop,
mouse mode uses PyAutoGUI for cursor control and X11 to detect Ctrl; install
PyAutoGUI with `python3 -m pip install pyautogui`.

## Run

```bash
python3 recunoastere_maini.py
```

Run this command from the project's Linux/WSL virtual environment, not from
Windows PowerShell or Command Prompt.

Choose a mode in the terminal:

- `t`: capture samples. Press `f` for fist, `p` for peace, `i` for index
  point, or `a` for palm.
- `r`: recognize gestures from saved samples.
- `c`: correction mode. Press the correct gesture key to add a labeled sample.
- `m`: mouse mode. A recognized right-hand index-point moves the pointer; a
  left-hand palm-to-fist transition or pressing Ctrl while the right
  index-point is detected clicks once. Pointer movement is lightly smoothed.
  In WSL, this mode starts `windows_mouse_bridge.ps1` to control the Windows
  cursor. Keep the gesture stable briefly to avoid accidental clicks; press
  `q` or Esc in the preview to stop.

Press `q` or Esc in the camera window to exit. Samples are stored in
`gesture_samples.csv` beside the script. The dataset may be deleted in case you
with to re-train your own set.

## Notes

The classifier uses up to 15 nearest landmark samples. Its displayed vote
agreement is not a calibrated probability. Rotation normalization and
distance-based rejection are heuristic and may need adjustment for different
hands and camera setups.
