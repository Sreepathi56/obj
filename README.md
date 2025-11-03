[README.md](https://github.com/user-attachments/files/23296373/README.md)
# Emotion Detection

A small Python project for detecting emotions using two approaches:
- DeepFace (real-time and image-based emotion analysis)
- FER (lightweight real-time emotion detection using MTCNN)

This repository contains a script that supports:
- Real-time webcam emotion detection (DeepFace)
- Emotion detection from a single image file (DeepFace)
- Multi-face emotion detection in an image (DeepFace)
- Real-time webcam emotion detection with FER (lighter alternative)

## Files
- `emotion_detection.py` (or the script file you provided) — main script with the four detection modes.

> If your script has a different name, update the examples below accordingly.

## Requirements

- Python 3.8+
- A webcam (for real-time detection)
- Internet connection the first time to download models used by DeepFace / FER

Recommended packages (example `requirements.txt`):
```
opencv-python
deepface
fer
mtcnn
tensorflow>=2.0  # or tensorflow-cpu if you don't have GPU
numpy
```

Note: Installing `deepface` may pull in `tensorflow` or `tensorflow-cpu`. If you have a GPU and want GPU-accelerated models, install the appropriate `tensorflow` package for your environment. On some systems, you may prefer `tensorflow-cpu` to avoid heavy GPU setup.

Install with pip:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
# or
pip install opencv-python deepface fer mtcnn tensorflow numpy
```

## Usage

Run the script and follow the interactive menu:

```bash
python emotion_detection.py
```

You will see:
```
=== EMOTION DETECTION PROGRAM ===

Choose an option:
1. Real-time emotion detection (DeepFace)
2. Detect emotion from image file
3. Detect multiple emotions in image
4. Real-time emotion detection (FER - lighter)
```

Examples:

- Real-time DeepFace:
  - Choose option `1`. The script opens your webcam, detects faces and runs DeepFace analysis every 5 frames to reduce overhead. Press `q` to quit.

- Analyze a single image with DeepFace:
  - Choose option `2` and enter the full path to the image file (e.g. `tests/photo.jpg`). The script will open a window with the annotated image and print the dominant emotion and scores to the console.

- Multiple faces in an image:
  - Choose option `3` and enter the image path. The script will detect faces using Haar cascades and analyze each face individually.

- Real-time FER:
  - Choose option `4`. FER uses MTCNN for face detection and is often lighter/faster on CPU for lower-throughput use cases. Press `q` to quit.

Notes:
- The script uses OpenCV's `imshow` so it must run in an environment with a display (not headless) unless you adapt it to save frames instead.
- If the webcam index is not `0` on your machine, change `cv2.VideoCapture(0)` to a different index or a video file path.

## Behavior & Parameters

- DeepFace is called with `enforce_detection=False` in the script; this avoids throwing exceptions when a face is not confidently detected but may produce unexpected outputs on non-face crops.
- For performance, the real-time DeepFace analysis runs only every 5 frames by default (change `if frame_count % 5 == 0:`).
- For more accurate face detection, you can swap Haar cascades for other detectors (MTCNN, RetinaFace). DeepFace also supports several backends.

## Troubleshooting

- "Could not open webcam":
  - Ensure no other application is using the camera.
  - On macOS grant Terminal/IDE camera permission in System Preferences / Security & Privacy.
  - Try a different camera index (1, 2, ...).

- DeepFace download / model load is slow on first run:
  - Models are downloaded the first time; subsequent runs are faster.
  - If you get space or permission issues, ensure your user can write to the cache directories or set the DeepFace cache folder via environment variables if needed.

- Errors from `fer` or `mtcnn`:
  - `mtcnn` needs proper installation; if you face errors, try re-installing with `pip install mtcnn`.
  - If MTCNN fails on some frames, consider falling back to Haar cascade detection.

- GPU Out Of Memory (OOM):
  - If you get OOM with TensorFlow on GPU, either install `tensorflow-cpu` or reduce frame size and frequency of model runs.

- No faces detected in images:
  - Try increasing `minSize`/adjusting `scaleFactor`/`minNeighbors` or using a different detector/backend.

## Performance tips

- Reduce video resolution: the script sets 640x480; lower it to 320x240 for faster processing.
- Increase the frame skip (e.g., analyze every 10th frame).
- Use a GPU-enabled TensorFlow installation for faster DeepFace inference.
- For many faces in a frame, consider batching face crops before analysis if moving to a custom inference pipeline.

## Extending the project

- Save results (timestamp, bounding boxes, emotion scores) to CSV/JSON or a database.
- Add logging and a CLI argument parser (argparse) to run non-interactively.
- Add a headless mode: write annotated frames to disk or stream via a web server (Flask/Streamlit).
- Replace Haar cascades with RetinaFace for improved detection accuracy in varied poses and lighting.

## Security & Privacy

- Be mindful of privacy and legal constraints when recording people or storing face/emotion data. Always get explicit consent from persons captured by the system, and ensure secure storage or anonymization if you retain results.

## License

MIT License — feel free to adapt and extend.

## Contact / Maintainers

- Maintainer: your GitHub user (add your contact info or repository-specific details here).

Enjoy experimenting with emotion detection! If you'd like, I can also:
- generate a recommended `requirements.txt`,
- add a minimal GitHub Actions workflow to run linting,
- or create a small example image and instructions for automated testing.
