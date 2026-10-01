# 🎯 YOLOv8 Real-Time Object Detection Showcase

A clean, modern computer vision portfolio project that demonstrates real-time (and near-real-time) object detection using **Ultralytics YOLOv8** and a polished **Streamlit** web interface.

This project is designed as a strong GitHub showcase for Machine Learning / Computer Vision Engineer roles.

---

## Features

- Load different YOLOv8 model sizes (Nano / Small / Medium)
- Adjustable confidence threshold
- **Image upload** with annotated results + detection table
- **Video upload** with full-frame processing
- **Webcam snapshot** support
- Clean, professional UI
- Modular code structure (`src/detector.py` for easy extension)
- Per-session temp directories that are cleaned up after video processing

---

## Project Structure

```
yolo-realtime-detector/
├── app.py                  # Main Streamlit application
├── run_cli.py              # Command-line detector for images / videos
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # + pytest for the test suite
├── .gitignore
├── README.md
├── src/
│   └── detector.py         # Reusable YOLO detector class + video helpers
└── tests/
    └── test_detector.py    # Unit tests (no model weights required)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/RetnuhFeel/yolo-realtime-detector.git
cd yolo-realtime-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# or
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The first time you run the app, Ultralytics will automatically download the selected YOLOv8 weights (`yolov8n.pt` etc.).

### 4. Launch the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Optional: ffmpeg

Ultralytics typically writes processed videos as `.avi`, which most browsers can't play inline. If `ffmpeg` is on your `PATH`, the app converts the result to H.264 `.mp4` automatically; otherwise it offers the file as a download.

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

### CLI

```bash
python run_cli.py --source path/to/image.jpg --save
python run_cli.py --source path/to/video.mp4 --conf 0.4
```

---

## How to Use

1. Select a model size in the sidebar (start with `yolov8n.pt` for speed).
2. Adjust the confidence threshold if desired.
3. Choose an input method:
   - **Image Upload** – best for quick testing and screenshots
   - **Video Upload** – processes every frame and returns an annotated video
   - **Webcam Snapshot** – take a photo and run detection immediately
4. Click the run button and view the annotated results + detection table.

---

## Next Steps / Ideas to Extend This Project

These are excellent ways to make the project even stronger for your resume:

- [ ] Add a **live continuous webcam** stream (using `streamlit-webrtc` or OpenCV + `st.image` loop)
- [ ] Fine-tune YOLOv8 on a custom dataset (e.g., tools, equipment, medical images, or domain-specific objects)
- [ ] Export detections to CSV / database
- [ ] Add object tracking (ByteTrack or BoT-SORT)
- [ ] Deploy the app publicly (Streamlit Community Cloud or Hugging Face Spaces)
- [ ] Add a simple performance comparison (FPS, mAP notes)
- [ ] Extend the CLI (`run_cli.py`) for batch processing of folders of images

---

## Tech Stack

- **Ultralytics YOLOv8** – state-of-the-art object detection
- **Streamlit** – rapid interactive UI
- **OpenCV** + **Pillow** – image handling
- **PyTorch** – underlying deep learning framework

---

## Author

**Hunter N. Leef**  
Computer Scientist • MS in Intelligent Systems & Machine Learning  
Portfolio project for remote ML / Computer Vision roles

---

## License

MIT License – feel free to use and modify.
