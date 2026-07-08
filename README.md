# Facial Meme Recogniser

A small OpenCV webcam app that detects a face, estimates the visible expression, and shows a meme that best matches it.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add memes

Put image files into folders named after expressions:

```text
memes/
  happy/
  surprised/
  dramatic/
  tired/
  neutral/
```

Supported image types are `.jpg`, `.jpeg`, `.png`, and `.webp`.

## Run

```powershell
python app.py
```

Use another camera if needed:

```powershell
python app.py --camera 1
```

Press `Q` to quit. Press `S` to save a screenshot.

## Notes

This first version uses OpenCV Haar cascades and simple image-processing heuristics. It is fast and easy to run, but it is not as accurate as a trained emotion-recognition model. The expression labels currently supported by the meme picker are:

- `happy`
- `surprised`
- `dramatic`
- `tired`
- `neutral`
