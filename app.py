"""
YOLOv8 Real-Time Object Detection Showcase
A clean Streamlit app demonstrating computer vision skills.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.detector import YOLODetector, convert_to_mp4, find_output_video

# Page config
st.set_page_config(
    page_title="YOLOv8 Object Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(model_name: str = "yolov8n.pt") -> YOLODetector:
    """Load the YOLO detector with caching so it only loads once per model."""
    return YOLODetector(model_name)


def run_image_detection(detector: YOLODetector, image: Image.Image, confidence: float, container):
    """Run detection on a PIL image and render the annotated result + table in ``container``."""
    try:
        with st.spinner("Detecting objects..."):
            results = detector.predict(source=np.array(image), conf=confidence)
            annotated_rgb = detector.annotate_rgb(results)
            df = detector.get_detections_df(results)
    except Exception as exc:  # noqa: BLE001 - surface any inference failure in the UI
        st.error(f"Detection failed: {exc}")
        return

    with container:
        st.subheader("Detections")
        st.image(annotated_rgb, caption="Annotated Result", use_container_width=True)
        if len(df) > 0:
            st.dataframe(df, use_container_width=True)
            st.success(f"Found {len(df)} object(s)")
        else:
            st.info("No objects detected above the confidence threshold.")


def run_video_detection(detector: YOLODetector, video_bytes: bytes, suffix: str, confidence: float, container):
    """Process an uploaded video in a private temp dir that is always cleaned up."""
    work_dir = Path(tempfile.mkdtemp(prefix="yolo_app_"))
    try:
        input_path = work_dir / f"input{suffix}"
        input_path.write_bytes(video_bytes)

        with st.spinner("Processing video frames... this may take a moment"):
            detector.predict(
                source=str(input_path),
                conf=confidence,
                save=True,
                project=str(work_dir),
                name="pred",
                exist_ok=True,
            )

        output = find_output_video(work_dir / "pred")
        with container:
            st.subheader("Processed Video")
            if output is None:
                st.warning("Could not locate the processed video file.")
                return

            if output.suffix.lower() != ".mp4":
                converted = convert_to_mp4(output)
                if converted is not None:
                    output = converted

            data = output.read_bytes()  # read before the temp dir is removed
            if output.suffix.lower() == ".mp4":
                st.video(data)
                st.success("Video processed successfully!")
            else:
                st.warning(
                    "The processed video is in .avi format, which most browsers can't play inline "
                    "(install ffmpeg to enable automatic conversion). Use the button below to download it."
                )
            st.download_button("Download processed video", data=data, file_name=output.name)
    except Exception as exc:  # noqa: BLE001 - surface any processing failure in the UI
        st.error(f"Video processing failed: {exc}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def open_image(uploaded) -> Image.Image | None:
    """Open an uploaded/captured image as RGB, showing an error if it is unreadable."""
    try:
        return Image.open(uploaded).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the image: {exc}")
        return None


def main():
    st.markdown('<p class="main-header">🎯 YOLOv8 Real-Time Object Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Computer Vision Showcase • Real-time detection with Ultralytics YOLOv8</p>',
        unsafe_allow_html=True
    )

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")

        model_option = st.selectbox(
            "Model Size",
            options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
            help="Nano is fastest. Small/Medium are more accurate but slower."
        )

        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05
        )

        st.markdown("---")
        st.markdown("**Input Source**")
        source_type = st.radio(
            "Choose input type",
            ["Image Upload", "Video Upload", "Webcam Snapshot"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### About this Project")
        st.markdown("""
        This project demonstrates:
        - Loading and running modern YOLO models
        - Image / video / webcam inference
        - Clean UI with Streamlit
        - Detection results in tabular form

        Built as a portfolio piece for ML/CV roles.
        """)

    # Load model
    try:
        detector = load_detector(model_option)
    except Exception as exc:  # noqa: BLE001 - e.g. download failure / corrupt weights
        st.error(f"Could not load model '{model_option}': {exc}")
        st.stop()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")

        if source_type == "Image Upload":
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if uploaded_file is not None:
                image = open_image(uploaded_file)
                if image is not None:
                    st.image(image, caption="Original Image", use_container_width=True)
                    if st.button("Run Detection", type="primary"):
                        run_image_detection(detector, image, confidence, col2)

        elif source_type == "Video Upload":
            uploaded_video = st.file_uploader(
                "Upload a video",
                type=["mp4", "avi", "mov", "mkv"]
            )
            if uploaded_video is not None:
                video_bytes = uploaded_video.getvalue()
                st.video(video_bytes)

                if st.button("Process Video", type="primary"):
                    suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
                    run_video_detection(detector, video_bytes, suffix, confidence, col2)

        elif source_type == "Webcam Snapshot":
            st.info("Click the camera button below to take a snapshot, then run detection.")
            camera_img = st.camera_input("Take a photo")

            if camera_img is not None:
                image = open_image(camera_img)
                if image is not None:
                    st.image(image, caption="Captured Image", use_container_width=True)
                    if st.button("Run Detection on Snapshot", type="primary"):
                        run_image_detection(detector, image, confidence, col2)

    # Footer
    st.markdown("---")
    st.caption("Built with Ultralytics YOLOv8 + Streamlit • Portfolio project for Computer Vision roles")


if __name__ == "__main__":
    main()
