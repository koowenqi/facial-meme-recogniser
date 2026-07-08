from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


APP_TITLE = "Facial Meme Recogniser"
SUPPORTED_MEME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PANEL_SIZE = (400, 800)
WINDOW_SIZE = (PANEL_SIZE[0] * 2, PANEL_SIZE[1])
CAMERA_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


@dataclass(frozen=True)
class ExpressionResult:
    label: str
    confidence: float
    details: str


class MemeLibrary:
    def __init__(self, meme_dir: Path) -> None:
        self.meme_dir = meme_dir
        self.memes = self._load_meme_paths()

    def _load_meme_paths(self) -> dict[str, list[Path]]:
        if not self.meme_dir.exists():
            return {}

        memes: dict[str, list[Path]] = {}
        for expression_dir in self.meme_dir.iterdir():
            if not expression_dir.is_dir():
                continue

            image_paths = [
                path
                for path in expression_dir.iterdir()
                if path.suffix.lower() in SUPPORTED_MEME_EXTENSIONS
            ]
            if image_paths:
                memes[expression_dir.name.lower()] = image_paths

        return memes

    def choose_for(self, expression: str) -> Path | None:
        exact_matches = self.memes.get(expression.lower())
        if exact_matches:
            return random.choice(exact_matches)

        fallback_matches = self.memes.get("neutral")
        if fallback_matches:
            return random.choice(fallback_matches)

        all_memes = [path for paths in self.memes.values() for path in paths]
        if all_memes:
            return random.choice(all_memes)

        return None


class ExpressionMatcher:
    def __init__(self) -> None:
        cascade_root = Path(cv2.data.haarcascades)
        self.face_cascade = self._load_cascade(cascade_root / "haarcascade_frontalface_default.xml")
        self.eye_cascade = self._load_cascade(cascade_root / "haarcascade_eye.xml")
        self.smile_cascade = self._load_cascade(cascade_root / "haarcascade_smile.xml")

    @staticmethod
    def _load_cascade(path: Path) -> cv2.CascadeClassifier:
        cascade_class = getattr(cv2, "CascadeClassifier", None)
        if cascade_class is None:
            raise RuntimeError(
                "This app needs OpenCV 4.x with Haar cascade support. "
                "Run: python -m pip install --force-reinstall -r requirements.txt"
            )

        if not path.exists():
            raise RuntimeError(
                f"OpenCV Haar cascade file is missing: {path}. "
                "Run: python -m pip install --force-reinstall -r requirements.txt"
            )

        cascade = cascade_class(str(path))
        if cascade.empty():
            raise RuntimeError(f"Could not load OpenCV cascade: {path}")
        return cascade

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = self._preprocess(frame)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
        )
        return sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)

    def classify_expression(self, frame: np.ndarray, face_rect: tuple[int, int, int, int]) -> ExpressionResult:
        gray = self._preprocess(frame)
        x, y, w, h = face_rect
        face = gray[y : y + h, x : x + w]

        upper_face = face[: int(h * 0.58), :]
        lower_face = face[int(h * 0.48) :, :]

        eyes = self.eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(18, 18),
        )
        smiles = self.smile_cascade.detectMultiScale(
            lower_face,
            scaleFactor=1.7,
            minNeighbors=18,
            minSize=(25, 18),
        )

        smile_score = self._largest_area_ratio(smiles, w * max(1, lower_face.shape[0]))
        eye_score = self._largest_area_ratio(eyes, w * max(1, upper_face.shape[0]))
        mouth_edge_score = self._edge_density(lower_face)
        contrast_score = float(np.std(face) / 128.0)

        if smile_score > 0.035:
            return ExpressionResult("happy", min(0.99, 0.55 + smile_score * 8), "strong smile detected")

        if len(eyes) >= 2 and eye_score > 0.055 and mouth_edge_score > 0.12:
            return ExpressionResult("surprised", min(0.95, 0.45 + eye_score * 4), "wide eyes and active mouth region")

        if contrast_score > 0.55 and mouth_edge_score > 0.11:
            return ExpressionResult("dramatic", min(0.9, 0.4 + contrast_score * 0.6), "high facial contrast and expression lines")

        if len(eyes) == 0 and contrast_score < 0.38:
            return ExpressionResult("tired", 0.62, "low contrast and no clear eye detection")

        return ExpressionResult("neutral", 0.55, "no strong expression cue found")

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    @staticmethod
    def _largest_area_ratio(rects: np.ndarray, containing_area: int) -> float:
        if len(rects) == 0 or containing_area <= 0:
            return 0.0
        largest_area = max(rect[2] * rect[3] for rect in rects)
        return float(largest_area / containing_area)

    @staticmethod
    def _edge_density(face_region: np.ndarray) -> float:
        if face_region.size == 0:
            return 0.0
        edges = cv2.Canny(face_region, 80, 160)
        return float(np.count_nonzero(edges) / edges.size)


def make_text_panel(lines: list[str], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 245, dtype=np.uint8)

    line_height = 42
    start_y = (height - (len(lines) - 1) * line_height) // 2
    for index, line in enumerate(lines):
        scale = 0.85 if len(line) <= 18 else 0.62
        thickness = 2
        text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = max(16, (width - text_size[0]) // 2)
        y = start_y + index * line_height
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), thickness)

    return panel


def load_meme_preview(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    if path is None:
        return make_text_panel(["Add meme images", "inside memes/<expression>/"], size)

    image = cv2.imread(str(path))
    if image is None:
        return load_meme_preview(None, size)

    return resize_to_fit(image, width, height)


def resize_to_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    img_height, img_width = image.shape[:2]
    scale = min(width / img_width, height / img_height)
    new_width = max(1, int(img_width * scale))
    new_height = max(1, int(img_height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def draw_result(
    frame: np.ndarray,
    face_rect: tuple[int, int, int, int],
    expression: ExpressionResult,
) -> None:
    x, y, w, h = face_rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 190, 110), 2)

    label = f"{expression.label} ({expression.confidence:.0%})"
    cv2.rectangle(frame, (x, max(0, y - 34)), (x + max(180, w), y), (36, 190, 110), -1)
    cv2.putText(frame, label, (x + 8, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def build_display(frame: np.ndarray, meme_preview: np.ndarray) -> np.ndarray:
    panel_width, panel_height = PANEL_SIZE
    frame_panel = resize_to_fit(frame, panel_width, panel_height)
    meme_panel = resize_to_fit(meme_preview, panel_width, panel_height)
    return np.hstack([frame_panel, meme_panel])


def open_camera(camera_index: int, backend_name: str) -> cv2.VideoCapture:
    backend = CAMERA_BACKENDS[backend_name]
    capture = cv2.VideoCapture(camera_index, backend)
    if capture.isOpened():
        return capture

    capture.release()
    raise RuntimeError(
        f"Could not open webcam at index {camera_index} with backend '{backend_name}'. "
        "Try another index or backend, for example: python app.py --camera 1 --backend msmf"
    )


def setup_window() -> None:
    cv2.namedWindow(APP_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(APP_TITLE, *WINDOW_SIZE)
    cv2.moveWindow(APP_TITLE, 0, 0)


def show_message_window(left_lines: list[str], right_lines: list[str]) -> None:
    setup_window()
    display = build_display(
        make_text_panel(left_lines, PANEL_SIZE),
        make_text_panel(right_lines, PANEL_SIZE),
    )
    cv2.imshow(APP_TITLE, display)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in {ord("q"), 27}:
            break
    cv2.destroyAllWindows()


def run(camera_index: int, meme_dir: Path, backend_name: str) -> None:
    matcher = ExpressionMatcher()
    library = MemeLibrary(meme_dir)
    capture = open_camera(camera_index, backend_name)

    setup_window()

    last_expression: ExpressionResult | None = None
    last_meme_path: Path | None = None
    stable_label = ""
    stable_count = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(
                f"Webcam at index {camera_index} opened, but no frames were received. "
                "Close other apps using the camera or try: python app.py --camera 1"
            )

        frame = cv2.flip(frame, 1)
        faces = matcher.detect_faces(frame)

        if faces:
            expression = matcher.classify_expression(frame, faces[0])
            draw_result(frame, faces[0], expression)

            if expression.label == stable_label:
                stable_count += 1
            else:
                stable_label = expression.label
                stable_count = 1

            if stable_count == 1 or expression.label != getattr(last_expression, "label", None):
                last_meme_path = library.choose_for(expression.label)

            last_expression = expression
        else:
            last_expression = None
            last_meme_path = None

        if last_expression is None:
            meme_preview = make_text_panel(["No face detected"], PANEL_SIZE)
        else:
            meme_preview = load_meme_preview(last_meme_path, PANEL_SIZE)

        display = build_display(frame, meme_preview)
        cv2.imshow(APP_TITLE, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite("meme_match_screenshot.jpg", display)

    capture.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect a face with OpenCV and match it to a meme folder.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to use. Default: 0")
    parser.add_argument(
        "--backend",
        choices=CAMERA_BACKENDS.keys(),
        default="dshow",
        help="OpenCV camera backend to use. Default: dshow",
    )
    parser.add_argument("--memes", type=Path, default=Path("memes"), help="Path to meme folders. Default: memes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(camera_index=args.camera, meme_dir=args.memes, backend_name=args.backend)
    except RuntimeError as error:
        print(error)
        show_message_window(["Webcam unavailable"], ["No face detected"])


if __name__ == "__main__":
    main()
