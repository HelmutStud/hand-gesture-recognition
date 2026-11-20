import csv
import math
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATASET_PATH = Path(__file__).with_name("gesture_samples.csv")
LEGACY_FEATURE_COLUMNS = [f"feature_{index}" for index in range(21 * 3)]
LEGACY_CSV_COLUMNS = ["gesture", *LEGACY_FEATURE_COLUMNS]
GESTURE_KEYS = {
    ord("f"): "fist",
    ord("p"): "peace",
    ord("i"): "index_point",
    ord("a"): "palm",
}
GESTURE_LABELS = tuple(GESTURE_KEYS.values())
FEATURE_COUNT = 21 * 3
FEATURE_COLUMNS = [f"rotation_normalized_{index}" for index in range(FEATURE_COUNT)]
CSV_COLUMNS = ["gesture", *FEATURE_COLUMNS]
FeatureVector = list[float]


class GestureClassifier:
    def __init__(self, neighbors: int = 15) -> None:
        self.neighbors = neighbors
        self.features: list[FeatureVector] = []
        self.labels: list[str] = []
        self.max_neighbor_distance: float | None = None

    def fit(
        self,
        features: Sequence[FeatureVector],
        labels: Sequence[str],
    ) -> None:
        if len(features) != len(labels):
            raise ValueError("Each training feature vector must have one label.")
        if not features:
            raise ValueError("No gesture samples are available for training.")
        if any(len(vector) != FEATURE_COUNT for vector in features):
            raise ValueError(f"Each sample must contain {FEATURE_COUNT} values.")
        if any(label not in GESTURE_LABELS for label in labels):
            raise ValueError("Training data contains an unknown gesture label.")

        self.features = [list(vector) for vector in features]
        self.labels = list(labels)
        if len(self.features) < 2:
            self.max_neighbor_distance = None
            return

        calibration_distances = []
        calibration_neighbors = min(self.neighbors, len(self.features) - 1)
        for index, sample in enumerate(self.features):
            distances = sorted(
                math.dist(sample, other)
                for other_index, other in enumerate(self.features)
                if other_index != index
            )
            calibration_distances.append(
                sum(distances[:calibration_neighbors]) / calibration_neighbors
            )

        calibration_distances.sort()
        percentile_index = math.ceil(0.95 * len(calibration_distances)) - 1
        self.max_neighbor_distance = calibration_distances[percentile_index]

    def predict(self, features: FeatureVector) -> tuple[str, float, float]:
        if not self.features:
            raise ValueError("The gesture classifier has not been trained.")
        if len(features) != FEATURE_COUNT:
            raise ValueError(f"Input must contain {FEATURE_COUNT} values.")

        neighbor_count = min(self.neighbors, len(self.features) - 1)
        if neighbor_count < 1 or self.max_neighbor_distance is None:
            return "unknown", 0.0, math.inf

        nearest = sorted(
            (
                (math.dist(features, sample), label, index)
                for index, (sample, label) in enumerate(
                    zip(self.features, self.labels)
                )
            ),
            key=lambda item: (item[0], item[2]),
        )[:neighbor_count]
        votes = Counter(label for _, label, _ in nearest)
        best_count = max(votes.values())
        best_label = next(label for _, label, _ in nearest if votes[label] == best_count)
        neighbor_agreement = best_count / len(nearest)
        mean_neighbor_distance = sum(distance for distance, _, _ in nearest) / len(nearest)
        if mean_neighbor_distance > self.max_neighbor_distance:
            best_label = "unknown"
        return best_label, neighbor_agreement, mean_neighbor_distance


def create_hand_landmarker():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Hand landmark model not found: {MODEL_PATH}\n"
            f"Download it from: {MODEL_URL}"
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hands(
    frame: Any,
    hands: Any,
    timestamp_ms: int,
) -> list[tuple[str, list[Any]]]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )
    results = hands.detect_for_video(mp_image, timestamp_ms)
    detected_hands = []
    for index, landmarks in enumerate(results.hand_landmarks):
        handedness = "?"
        if index < len(results.handedness) and results.handedness[index]:
            category_name = results.handedness[index][0].category_name
            if category_name:
                handedness = {"L": "R", "R": "L"}.get(
                    category_name[0].upper(),
                    "?",
                )
        detected_hands.append((handedness, landmarks))
    return detected_hands


def extract_features(landmarks: Sequence[Any]) -> FeatureVector:
    if len(landmarks) != 21:
        raise ValueError("A hand sample must contain exactly 21 landmarks.")

    wrist = landmarks[0]
    scale = max(
        math.hypot(point.x - wrist.x, point.y - wrist.y)
        for point in landmarks
    )
    if scale < 1e-6:
        raise ValueError("Could not normalize hand landmarks with zero hand size.")

    features = [
        coordinate
        for point in landmarks
        for coordinate in (
            (point.x - wrist.x) / scale,
            (point.y - wrist.y) / scale,
            (point.z - wrist.z) / scale,
        )
    ]
    return normalize_rotation(features)


def normalize_rotation(features: FeatureVector) -> FeatureVector:
    middle_base_x = features[9 * 3]
    middle_base_y = features[9 * 3 + 1]
    direction_length = math.hypot(middle_base_x, middle_base_y)
    if direction_length < 1e-6:
        raise ValueError("Could not determine hand rotation from its palm landmarks.")

    angle = -math.pi / 2 - math.atan2(middle_base_y, middle_base_x)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    normalized: FeatureVector = []
    for index in range(0, len(features), 3):
        x, y, z = features[index:index + 3]
        normalized.extend((x * cosine - y * sine, x * sine + y * cosine, z))
    return normalized


def rewrite_samples(features: Sequence[FeatureVector], labels: Sequence[str]) -> None:
    backup_path = DATASET_PATH.with_name("gesture_samples_before_rotation.csv")
    if not backup_path.exists():
        shutil.copy2(DATASET_PATH, backup_path)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=DATASET_PATH.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for label, vector in zip(labels, features):
                row = {"gesture": label}
                row.update(dict(zip(FEATURE_COLUMNS, vector)))
                writer.writerow(row)
        temporary_path.replace(DATASET_PATH)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_samples() -> tuple[list[FeatureVector], list[str]]:
    if not DATASET_PATH.exists():
        return [], []

    features: list[FeatureVector] = []
    labels: list[str] = []
    migrate_legacy_samples = False
    with DATASET_PATH.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if reader.fieldnames == LEGACY_CSV_COLUMNS:
            migrate_legacy_samples = True
            feature_columns = LEGACY_FEATURE_COLUMNS
        elif reader.fieldnames == CSV_COLUMNS:
            feature_columns = FEATURE_COLUMNS
        else:
            raise ValueError(
                f"{DATASET_PATH} has an invalid header. "
                "Remove or rename it to start a fresh dataset."
            )

        for row_number, row in enumerate(reader, start=2):
            label = row["gesture"]
            if label not in GESTURE_LABELS:
                raise ValueError(
                    f"Unknown gesture label on row {row_number}: {label!r}"
                )
            try:
                vector = [float(row[column]) for column in feature_columns]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid feature value on row {row_number} of {DATASET_PATH}."
                ) from error
            if migrate_legacy_samples:
                vector = normalize_rotation(vector)
            features.append(vector)
            labels.append(label)

    if migrate_legacy_samples:
        rewrite_samples(features, labels)

    return features, labels


def save_sample(features: FeatureVector, label: str) -> None:
    write_header = not DATASET_PATH.exists() or DATASET_PATH.stat().st_size == 0
    with DATASET_PATH.open("a", newline="", encoding="utf-8") as dataset_file:
        writer = csv.DictWriter(dataset_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        row = {"gesture": label}
        row.update(
            {
                column: value
                for column, value in zip(FEATURE_COLUMNS, features)
            }
        )
        writer.writerow(row)


def draw_hand_landmarks(
    frame: Any,
    detected_hands: Sequence[tuple[str, Sequence[Any]]],
) -> None:
    for hand_label, landmarks in detected_hands:
        mp.tasks.vision.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
        )


def select_mode() -> str:
    while True:
        choice = input(
            "Choose mode: [t]rain, [r]ecognize, or [c]orrect predictions: "
        ).strip().lower()
        if choice in ("t", "r", "c"):
            return choice
        print("Please enter 't', 'r', or 'c'.")


def run_camera(
    mode: str,
    features: list[FeatureVector],
    labels: list[str],
    classifier: GestureClassifier,
) -> None:
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the camera. Check that it is connected.")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        with create_hand_landmarker() as hands:
            last_timestamp_ms = -1
            last_capture_time = 0.0
            if mode == "c":
                status = "Correct: press f=fist, p=peace, i=index, a=palm"
            else:
                status = "Show a hand; press f=fist, p=peace, i=index, a=palm"

            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the camera.")

                frame = cv2.flip(frame, 1)
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms
                detected_hands = detect_hands(frame, hands, timestamp_ms)
                draw_hand_landmarks(frame, detected_hands)

                for hand_label, landmarks in detected_hands:
                    if mode in ("r", "c"):
                        label = "unknown"
                        if classifier.features:
                            label, neighbor_agreement, distance = classifier.predict(
                                extract_features(landmarks)
                            )
                            if label == "unknown":
                                label_text = (
                                    f"unknown (far; {distance:.2f} distance)"
                                    if math.isfinite(distance)
                                    else "unknown (need more samples)"
                                )
                            else:
                                label_text = (
                                    f"{label} ({neighbor_agreement:.0%} vote)"
                                )
                        else:
                            label_text = "unknown"
                        label_text = f"[{hand_label}] {label_text}"
                    else:
                        label_text = f"[{hand_label}]"

                    wrist = landmarks[0]
                    position = (
                        int(wrist.x * frame.shape[1]),
                        max(25, int(wrist.y * frame.shape[0]) - 40),
                    )
                    cv2.putText(
                        frame,
                        label_text,
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                if mode in ("t", "c"):
                    counts = Counter(labels)
                    count_text = "  ".join(
                        f"{label}: {counts[label]}" for label in GESTURE_LABELS
                    )
                    cv2.putText(
                        frame,
                        f"Samples  {count_text}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    status,
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Camera", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

                if mode in ("t", "c") and key in GESTURE_KEYS:
                    now = time.monotonic()
                    if detected_hands and now - last_capture_time >= 0.5:
                        label = GESTURE_KEYS[key]
                        sample = extract_features(detected_hands[0][1])
                        save_sample(sample, label)
                        features.append(sample)
                        labels.append(label)
                        classifier.fit(features, labels)
                        last_capture_time = now
                        if mode == "c":
                            status = (
                                f"Corrected to {label}; saved sample "
                                f"#{Counter(labels)[label]}"
                            )
                        else:
                            status = (
                                f"Saved {label} sample #{Counter(labels)[label]}; "
                                "vary hand angle and position"
                            )
                    elif not detected_hands:
                        status = (
                            "No hand detected; show your hand, then press a label key"
                        )
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    mode = select_mode()
    features, labels = load_samples()
    classifier = GestureClassifier()

    if mode in ("r", "c"):
        if features:
            classifier.fit(features, labels)
            print(f"Loaded {len(labels)} training samples from {DATASET_PATH}.")
        else:
            print(
                "No training data found. Detected hands will be labeled unknown; "
                "press a gesture key to add a correction sample."
            )
    if mode == "t":
        print(
            "Training mode: tap f for fist, p for peace, "
            "i for index-point, a for palm."
        )
        print("Press each key repeatedly with small changes in hand angle/position.")
        if labels:
            print(f"Loaded {len(labels)} existing samples; new samples will be added.")
    elif mode == "c":
        print(
            "Correction mode: if a prediction is wrong, press the correct "
            "gesture key (f/p/i/a) to save that hand as a labeled sample."
        )

    run_camera(mode, features, labels, classifier)

    if mode in ("t", "c"):
        counts = Counter(labels)
        print(f"Dataset now contains {len(labels)} total samples at {DATASET_PATH}.")
        print("Samples by gesture: " + ", ".join(
            f"{label}={counts[label]}" for label in GESTURE_LABELS
        ))


if __name__ == "__main__":
    main()
