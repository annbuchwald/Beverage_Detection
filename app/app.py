import streamlit as st
from PIL import Image
from ultralytics import YOLO

from helpers import create_detection_dataframe, detect_objects, draw_boxes

global MODEL


@st.cache_resource
def load_model() -> YOLO:
    return YOLO("app/data/best_weights.pt")


MODEL = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider(
    "Choose your desired confidence threshold!",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        result = detect_objects(model=MODEL, image=image, conf_threshold=conf_threshold)

        image_with_boxes = draw_boxes(image.copy(), result)
        st.image(image_with_boxes, caption="Detection Result", use_column_width=True)

        st.subheader("Detection Details:")
        df_details = create_detection_dataframe(result)
        st.write(f"Found {len(result.boxes.cls)} items in total!")
        st.dataframe(df_details, hide_index=True)
