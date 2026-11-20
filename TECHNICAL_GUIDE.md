# Technical Guide: Hand Gesture Recognition

This guide explains the implementation in [`recunoastere_maini.py`](./recunoastere_maini.py)
and the Windows input helper, [`windows_mouse_bridge.ps1`](./windows_mouse_bridge.ps1).
It describes the current code paths and runtime behavior; it is not a MediaPipe
or OpenCV API reference.

## 1. System overview

The application uses a single camera loop for training, recognition,
correction, and mouse control:

```text
Camera frame
    |
    v
Mirror frame -> MediaPipe Hand Landmarker -> handedness + 21 landmarks
                                              |
                                              v
                              wrist-relative feature extraction
                              + rotation normalization
                                              |
                                  k-nearest-neighbors
                                              |
                 +----------------------------+-----------------------+
                 |                            |                       |
           draw labels                 capture/save sample       mouse mode
                                                                  |
                                             index fingertip -> smoothed cursor
                                             Ctrl press / palm-to-fist -> click

Static image
    |
    v
MediaPipe IMAGE mode -> hand landmarks -> skeleton overlay -> output image
```

In mouse mode under WSL, gesture recognition stays in Linux. A long-running
PowerShell child process receives simple cursor commands and forwards them to
the Windows desktop using Windows APIs. On a regular Linux desktop, PyAutoGUI
controls the pointer and X11 is queried for the Ctrl key state.

## 2. Imports, constants, and shared state

The standard-library imports provide CSV persistence, distance/math
calculations, paths, temporary-file replacement, process communication,
threading, timing, and vote counting. OpenCV (`cv2`) captures and displays
frames. MediaPipe Tasks (`mediapipe`) detects hands and draws their landmarks.

Important constants near the top of the Python file:

| Name | Purpose |
| --- | --- |
| `MOUSE_GESTURE_STABLE_SECONDS` | How long a palm or fist must remain the candidate gesture before the gesture click logic accepts it. |
| `MOUSE_CLICK_COOLDOWN_SECONDS` | Minimum time between clicks from the supported click actions. |
| `MOUSE_SMOOTHING_ALPHA` | Fraction of the remaining cursor distance applied each frame. `0.45` means each move covers 45% of the difference from the previous smoothed position. |
| `MODEL_PATH` / `MODEL_URL` | Local MediaPipe model location and download address shown when the file is missing. |
| `DATASET_PATH` | CSV file used for saved gesture examples. |
| `GESTURE_KEYS` | Keyboard-to-label mapping: `f` fist, `p` peace, `i` index point, `a` palm. |
| `FEATURE_COUNT` | Number of values in one normalized sample: 21 landmarks times 3 coordinates. |
| `CSV_COLUMNS` | Dataset label column followed by the normalized feature columns. |

`ClickGestureState` stores the candidate and stable left-hand click gestures,
whether a palm has armed the next fist click, the last click time, and whether
Ctrl was pressed on the previous frame. The previous-key state lets Ctrl
trigger on a press transition rather than repeatedly while held.

`MouseController` groups the selected mouse backend, screen dimensions, key
monitor, and smoothed cursor coordinates. Its `close` method releases the
backend and key-monitor resources when mouse mode ends.

## 3. Hand detection and feature preparation

### `create_hand_landmarker`

This function checks that `hand_landmarker.task` exists and creates a MediaPipe
Tasks `HandLandmarker` in video mode. It is configured to track up to two hands
and to use the configured detection and tracking confidence thresholds. Video
mode requires a strictly increasing timestamp for each submitted frame.

### `detect_hands`

This function converts an OpenCV BGR frame to RGB, wraps it as a MediaPipe
image, and calls `detect_for_video`. It returns each hand's landmarks alongside
a short handedness label.

The camera image is mirrored before detection in `run_camera`. The returned
MediaPipe left/right label is therefore swapped here so the displayed labels
and mouse-mode left/right behavior match the mirrored preview.

### `detect_hands_in_image`

This helper converts a still BGR image to an RGB MediaPipe image and calls the
landmarker's synchronous `detect` method. It returns the landmark lists in the
same container shape used by the shared skeleton-drawing helper. It does not
extract gesture features or run the classifier.

### `extract_features`

MediaPipe returns 21 landmarks per hand. Each landmark has normalized `x`, `y`,
and `z` coordinates. Feature extraction:

1. Uses landmark 0 (the wrist) as the origin.
2. Finds a hand-size scale from the largest 2D wrist-to-landmark distance.
3. Subtracts wrist coordinates and divides by that scale.
4. Sends the resulting 63 values to `normalize_rotation`.

The resulting feature values describe the hand shape relative to its wrist,
rather than the hand's absolute camera position or apparent size.

### `normalize_rotation`

This helper uses landmark 9 (the middle-finger base) to estimate the palm's
orientation in the image plane. It rotates each landmark's `x` and `y` values
so that this direction is aligned consistently. The `z` value is retained.
Invalid or degenerate landmarks raise a `ValueError` rather than producing a
misleading feature vector.

## 4. Gesture classifier

`GestureClassifier` implements a small k-nearest-neighbors classifier using
Euclidean distance (`math.dist`) over the 63-value vectors.

### `fit(features, labels)`

`fit` validates the sample/label counts, vector lengths, and gesture labels,
then keeps copies of the training data. It also calibrates a distance
rejection limit: for each training item it calculates the mean distance to its
nearest other items, then uses the 95th-percentile value as the limit.

### `predict(features)`

For an input feature vector, `predict`:

1. Finds the nearest training samples.
2. Counts their gesture labels and chooses the majority label.
3. Returns the winning label, neighbor vote agreement, and mean neighbor
   distance.
4. Returns `unknown` when the mean distance exceeds the calibrated limit.

With fewer than two training examples, the classifier cannot produce a
meaningful neighbor set and returns `unknown`. Vote agreement is a neighbor
vote ratio, not a calibrated probability.

## 5. Dataset storage and migration

The dataset is a CSV file. Each row has a `gesture` label and 63
rotation-normalized numeric feature values.

- `load_samples` reads and validates the file. It accepts both the current
  normalized header and the older `feature_*` header.
- If an older file is found, the features are rotation-normalized and
  `rewrite_samples` writes the converted file.
- `rewrite_samples` creates a one-time backup if needed, writes a temporary
  CSV in the dataset directory, and replaces the dataset with the completed
  file.
- `save_sample` appends one current-format sample, creating the header when
  the dataset is new or empty.

Invalid headers, labels, or numeric values produce explicit errors identifying
the problem row/file.

## 6. Mouse control

### Backend selection: `create_mouse_controller`

When `WSL_INTEROP` is present, the function starts `WindowsMouseBridge` and
uses the Windows screen dimensions reported by the helper. Otherwise it
initializes PyAutoGUI and a `CtrlKeyMonitor` for a Linux X11 desktop.

The Windows bridge uses these line-based commands:

| Command | Operation |
| --- | --- |
| `M x y` | Move the Windows cursor to screen coordinates `x, y`. |
| `C` | Send a left mouse-button down and up event. |
| `K` | Report whether Ctrl is currently pressed as `CTRL|0` or `CTRL|1`. |
| `Q` | Exit the helper cleanly. |

At startup the helper reports `READY|width|height`. The Python side checks the
response, dimensions, child-process state, and write errors.

`WindowsMouseBridge` has a background thread that polls Ctrl approximately
every 30 milliseconds. A command lock keeps polling commands, cursor commands,
and shutdown writes from interleaving. The state lock protects the latest
Ctrl value and any polling error.

On Linux, `CtrlKeyMonitor` resolves the X11 keycodes for left and right Ctrl
and checks the X server's current key map. Its X display connection is closed
when the mouse controller is closed.

### Cursor mapping and smoothing: `move_mouse_to_index`

Mouse movement is driven by landmark 8, the index fingertip, from the
recognized right-hand `index_point`. Its normalized camera coordinates are
scaled to the Windows or Linux screen dimensions and clamped to valid screen
positions.

The smoothing is an exponential moving average:

```text
smoothed_next = smoothed_previous
                + alpha * (target - smoothed_previous)
```

With `MOUSE_SMOOTHING_ALPHA = 0.45`, a move covers 45% of the remaining
distance each frame. The first detected position initializes the smoothed
coordinates directly so the cursor starts at the hand rather than drifting
from an arbitrary origin.

### Click actions

`update_ctrl_click` detects a Ctrl press edge (`pressed` now, not pressed on
the previous frame). It clicks only if the right hand is detected and
classified as `index_point` in that frame. Holding Ctrl does not repeat the
click. It also respects the shared click cooldown.

`update_click_gesture` handles the other click action. The left hand must show
a stable palm and then a stable fist. A fist held in place does not repeat;
the next click requires a new stable palm-to-fist transition. Missing or
unrecognized left-hand gestures reset its candidate state.

## 7. Camera loop and modes

### `run_camera`

`run_camera` opens camera device 0 using OpenCV's V4L2 backend, requests MJPG
at 640x480, and owns the frame loop. Each iteration:

1. Reads and horizontally mirrors one frame.
2. Generates a monotonically increasing timestamp.
3. Runs hand detection once and draws the hand skeletons.
4. In recognition, correction, and mouse modes, extracts features and predicts
   each detected hand once.
5. Draws labels and mode status.
6. In mouse mode, moves the pointer and updates both click mechanisms.
7. Shows the preview, polls its keyboard, and handles training/correction keys.

Press `q` or Escape in the preview to exit. The `finally` block releases the
camera, closes OpenCV windows, and closes mouse/key-monitor resources.

### `run_static_image_mode`

Static-image mode prompts for an input path and an optional output path. When
the output is left blank, it writes a sibling file named
`<input-stem>_hands<input-extension>` (or `.jpg` if the input has no extension).
`resolve_image_path` accepts Linux paths and, under WSL, Windows drive paths;
it uses `wslpath -u` to convert a Windows path such as `D:\Images\1.jpg` into
its mounted Linux path before OpenCV opens or writes the file.
It opens the image with OpenCV, creates a MediaPipe Hand Landmarker in
`RunningMode.IMAGE`, detects up to two hands, draws the detected skeletons,
and saves the annotated copy. It rejects a missing/unreadable input and
prevents the output path from replacing the source image.
If MediaPipe detects no hands, the output is an unchanged copy and the console
reports that no skeletons were detected. The bundled model is designed for
human hands and does not provide general robotic-hand detection.

This path deliberately skips handedness labels, gesture feature extraction,
classification, the webcam loop, and training-dataset loading.

### `select_mode` and `main`

`select_mode` prompts for one of five modes:

| Mode | Behavior |
| --- | --- |
| `t` | Training: capture a labeled sample when a gesture key is pressed. |
| `r` | Recognition: classify and display each detected hand. |
| `c` | Correction: classify hands and append a sample with the corrected label. |
| `m` | Mouse: recognized right index-point moves the pointer; Ctrl plus that hand clicks; left palm-to-fist clicks. |
| `s` | Static image: detect hands in one image and save a skeleton-overlay copy without gesture recognition. |

`main` routes static-image mode directly to `run_static_image_mode`, so it
does not load the CSV dataset or open a camera. The other modes load saved
samples once, fit the classifier when needed, print mode-specific instructions,
and start `run_camera`. In training/correction modes it prints updated sample
totals after the camera loop exits.

## 8. Runtime requirements and limitations

- The camera is opened through V4L2, so the current script targets Linux/WSL;
  native Windows camera use is not implemented.
- Under WSL, Windows cursor control requires Windows interop, `powershell.exe`
  on the WSL `PATH`, and the companion PowerShell file beside the Python
  script.
- Linux desktop mouse control requires PyAutoGUI and an X11 session. Ctrl
  detection uses X11 key state.
- WSL camera use requires the webcam to be attached to WSL, and the OpenCV
  preview requires WSLg or another supported GUI.
- The MediaPipe model is a separate local file. The dataset is separate from
  the source code and is not recreated automatically when absent; train or
  supply samples before expecting known classifications.
- Static-image mode still requires the MediaPipe model file, but does not
  require a camera or a gesture dataset.
- Cursor movement is smoothed and therefore intentionally trails the fingertip
  slightly. Increase `MOUSE_SMOOTHING_ALPHA` for less smoothing and faster
  response; decrease it for steadier but slower movement.

For installation and webcam setup, see the project [README](./README.md).
