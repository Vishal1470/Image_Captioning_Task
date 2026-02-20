import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Image Captioning App",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Captioning App")
st.caption("Powered by BLIP (PyTorch + HuggingFace)")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model.eval()
    return processor, model


processor, model = load_model()

# -----------------------------
# Upload image
# -----------------------------
uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=None)


    if st.button("Generate Caption 🚀"):
        with st.spinner("Please wait... Generating caption"):
            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=40
                )

            caption = processor.decode(
                output[0],
                skip_special_tokens=True
            )

        st.success("Done!")
        st.subheader("📝 Generated Caption")
        st.write(caption)
