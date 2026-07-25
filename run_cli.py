"""
Simple command-line interface for quick testing of YOLO detection.
Usage:
    python run_cli.py --source path/to/image.jpg
    python run_cli.py --source path/to/video.mp4 --conf 0.4
"""

import argparse
from ultralytics import YOLO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 CLI Detector")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model name or path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save annotated results")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        verbose=True
    )

    print(f"\nProcessed: {args.source}")
    print(f"Model: {args.model} | Confidence: {args.conf}")
    if args.save:
        print("Results saved to the 'runs/detect' folder.")


if __name__ == "__main__":
    main()
