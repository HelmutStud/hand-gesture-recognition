# Hand Gesture Recognition

A webcam hand-tracking and gesture-recognition project using OpenCV and
MediaPipe Tasks. It can collect labeled hand-landmark samples, recognize
gestures with a k-nearest-neighbors classifier, and add corrections to the
training data.

## Requirements

- Python 3.10 or later
- A working webcam
- Linux/WSL with GUI support for the OpenCV preview window

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

## Run

```bash
python3 recunoastere_maini.py
```

Choose a mode in the terminal:

- `t`: capture samples. Press `f` for fist, `p` for peace, `i` for index
  point, or `a` for palm.
- `r`: recognize gestures from saved samples.
- `c`: correction mode. Press the correct gesture key to add a labeled sample.

Press `q` or Esc in the camera window to exit. Samples are stored in
`gesture_samples.csv` beside the script. The dataset, migration backup, model,
and legacy script backup are ignored by Git, so personal training data and
downloaded model files are not pushed by default.

## Notes

The classifier uses up to 15 nearest landmark samples. Its displayed vote
agreement is not a calibrated probability. Rotation normalization and
distance-based rejection are heuristic and may need adjustment for different
hands and camera setups.
