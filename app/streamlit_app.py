from io import BytesIO

import config
import httpx

import streamlit as st
from PIL import Image

from config import CLASSES


def display_label(name):
    return "No Tumor" if name == "notumor" else name


st.set_page_config(page_title="Brain MRI Classifier", page_icon="🧠")
st.title("🧠 Brain MRI Classifier")
st.caption("Educational demo — not a medical diagnosis.")

# Both image sources use the same preview and prediction below.
source = st.radio("Image source", ["Upload image", "Choose example"], horizontal=True)
image_source = None
if source == "Upload image":
    image_source = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])
else:
    examples = []
    for name in CLASSES:
        folder = config.EXAMPLES_DIR / name
        examples.extend(sorted(folder.glob("*.jpg"))[:3])
    if examples:
        image_source = st.selectbox("Example", examples, format_func=lambda p: f"{display_label(p.parent.name)} / {p.name}")
    else:
        st.info("No example images available.")

if image_source is not None:
    try:
        with Image.open(image_source) as image:
            image = image.convert("RGB")
    except (OSError, ValueError):
        st.error("Please choose a readable image.")
        st.stop()
    st.image(image, width="stretch")
    # Run inference only when requested, not on every interface update.
    if st.button("Predict", type="primary"):
        # Send the displayed image to the API, for uploads and examples alike.
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        try:
            with st.spinner("Running inference..."):
                response = httpx.post(
                    f"{config.API_URL.rstrip('/')}/predict",
                    files={"file": ("image.png", buffer.getvalue(), "image/png")},
                    timeout=30,
                )
                response.raise_for_status()
        except httpx.RequestError:
            st.error("Cannot reach the API. Check that it is running and try again.")
        except httpx.HTTPStatusError as error:
            st.error(f"Prediction failed: HTTP {error.response.status_code}")
        else:
            result = response.json()
            st.success(f"Prediction: **{display_label(result['predicted_class'])}**")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.bar_chart(
                {display_label(name): score for name, score in result["probabilities"].items()},
                horizontal=True,
            )
