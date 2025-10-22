import os
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
EXPERIMENTS_DIR = Path(__file__).parent
if ROBOFLOW_API_KEY is None:
    raise RuntimeError(
        "`ROBOFLOW_API_KEY` not found in the environment! Please set "
        "`ROBOFLOW_API_KEY` to your API key to connect with Roboflow!"
    )


def train():
    """Run YOLOv8 fine-tuning on the Bevarage Containers Dataset."""
    # Roboflow setup
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    dataset = rf.workspace("roboflow-universe-projects").project("beverage-containers-3atxb").version(3)
    dataset = dataset.download("yolov5", location=str(EXPERIMENTS_DIR / "dataset"))
    # Load a pretrained YOLO model
    model = YOLO(str(EXPERIMENTS_DIR / "yolov8n.pt"))
    # Fine-tune the model
    train_config = {
        "data": str(EXPERIMENTS_DIR / "dataset" / "data.yaml"),
        "epochs": 100,
        "imgsz": 640,
    }
    model.train(**train_config)


if __name__ == "__main__":
    train()
