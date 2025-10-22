import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.engine.results import Results


def detect_objects(model: YOLO, image: Image.Image, conf_threshold: float) -> Results:
    """Perform object detection on an image using a YOLO model.

    Args:
        model (YOLO): The YOLO model to use for detection.
        image (Image.Image): The input image on which to perform detection.
        conf_threshold (float): The confidence threshold for detections.

    Returns:
        Results: A Results object containing detection information.
    """
    return model(image, conf=conf_threshold)[0]


def draw_boxes(image: Image.Image, result: Results) -> Image.Image:
    """Draw bounding boxes and labels on an image based on detection results.

    Args:
        image (Image.Image): The input image on which to draw boxes.
        result (Results): The detection results from a YOLO model.

    Returns:
        Image.Image: The input image with bounding boxes and labels drawn.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("app/data/base_font.ttf", 40)

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    names = result.names

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        label = names[int(cls)]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Calculate text size and position using textbbox
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (x1, y1 - text_height - 5)

        # Draw a filled rectangle behind the text for better visibility
        draw.rectangle(
            [
                text_position[0],
                text_position[1],
                text_position[0] + text_width,
                text_position[1] + text_height,
            ],
            fill="red",
        )
        draw.text(text_position, label, fill="white", font=font)

    return image


def create_detection_dataframe(result: Results) -> pd.DataFrame:
    """Create a pandas DataFrame from YOLO detection results.

    Args:
        result (Results): The detection results from a YOLO model.

    Returns:
        pd.DataFrame: A DataFrame containing detection information.
        The DataFrame has columns for Class, Confidence, and Bounding Box.
    """
    detection_data = []
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        detection_data.append(
            {
                "Class": result.names[int(cls)],
                "Confidence": f"{conf:.2f}",
                "Bounding Box (XYXY format)": box.numpy().round(),
            }
        )
    return pd.DataFrame(detection_data)
