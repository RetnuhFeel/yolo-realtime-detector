"""
YOLOv8 Real-Time Object Detection Showcase
A clean Streamlit app demonstrating computer vision skills.
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

# Optional: import our wrapper if you want cleaner separation
# from src.detector import YOLODetector

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
def load_model(model_name: str = "yolov8n.pt"):
    """Load YOLO model with caching so it only loads once."""
    model = YOLO(model_name)
    return model


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
    model = load_model(model_option)

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
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Original Image", use_container_width=True)

                if st.button("Run Detection", type="primary"):
                    with st.spinner("Detecting objects..."):
                        results = model.predict(
                            source=np.array(image),
                            conf=confidence,
                            verbose=False
                        )
                        annotated = results[0].plot()  # BGR
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                        with col2:
                            st.subheader("Detections")
                            st.image(annotated_rgb, caption="Annotated Result", use_container_width=True)

                            # Detection table
                            boxes = results[0].boxes
                            if boxes is not None and len(boxes) > 0:
                                data = []
                                for box in boxes:
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    name = model.names[cls_id]
                                    data.append({
                                        "Class": name,
                                        "Confidence": f"{conf:.2f}",
                                    })
                                st.dataframe(data, use_container_width=True)
                                st.success(f"Found {len(data)} object(s)")
                            else:
                                st.info("No objects detected above the confidence threshold.")

        elif source_type == "Video Upload":
            uploaded_video = st.file_uploader(
                "Upload a video",
                type=["mp4", "avi", "mov", "mkv"]
            )
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.close()

                st.video(tfile.name)

                if st.button("Process Video", type="primary"):
                    with st.spinner("Processing video frames... this may take a moment"):
                        # Run prediction on video (Ultralytics handles it)
                        results = model.predict(
                            source=tfile.name,
                            conf=confidence,
                            save=True,
                            project="outputs",
                            name="video_pred",
                            exist_ok=True,
                            verbose=False
                        )

                        # Find the output video
                        output_dir = Path("outputs/video_pred")
                        output_videos = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.avi"))

                        with col2:
                            st.subheader("Processed Video")
                            if output_videos:
                                st.video(str(output_videos[0]))
                                st.success("Video processed successfully!")
                            else:
                                st.warning("Could not locate the processed video file.")

                        # Clean up temp input
                        os.unlink(tfile.name)

        elif source_type == "Webcam Snapshot":
            st.info("Click the camera button below to take a snapshot, then run detection.")
            camera_img = st.camera_input("Take a photo")

            if camera_img is not None:
                image = Image.open(camera_img).convert("RGB")
                st.image(image, caption="Captured Image", use_container_width=True)

                if st.button("Run Detection on Snapshot", type="primary"):
                    with st.spinner("Detecting..."):
                        results = model.predict(
                            source=np.array(image),
                            conf=confidence,
                            verbose=False
                        )
                        annotated = results[0].plot()
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                        with col2:
                            st.subheader("Detections")
                            st.image(annotated_rgb, caption="Annotated Result", use_container_width=True)

                            boxes = results[0].boxes
                            if boxes is not None and len(boxes) > 0:
                                data = []
                                for box in boxes:
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    name = model.names[cls_id]
                                    data.append({"Class": name, "Confidence": f"{conf:.2f}"})
                                st.dataframe(data, use_container_width=True)
                            else:
                                st.info("No objects detected.")

    # Footer
    st.markdown("---")
    st.caption("Built with Ultralytics YOLOv8 + Streamlit • Portfolio project for Computer Vision roles")


if __name__ == "__main__":
    main()
