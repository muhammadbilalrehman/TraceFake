import streamlit as st
import tensorflow as tf
import numpy as np
import exifread
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="TraceFake AI", page_icon="🛡️")

# 2. Load Model (Cached to prevent reloading on every click)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "tracefake_resave.keras",
        compile=False
    )

model = load_model()

# 3. EXIF Extraction Function
def get_metadata(file):
    try:
        tags = exifread.process_file(file, details=False)
        return {
            "Make": tags.get("Image Make", "N/A"),
            "Model": tags.get("Image Model", "N/A"),
            "Software": tags.get("Image Software", "N/A"),
            "Date": tags.get("EXIF DateTimeOriginal", "N/A")
        }
    except:
        return None

# 4. Streamlit UI
st.title("🛡️ TraceFake: AI Image Verifier")
st.write("Upload a face image to detect if it is **Real** or **AI-Generated**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")
    
    # Process and Predict
    with st.spinner('Analyzing Pixels...'):
        # Preprocessing matching your training logic
        test_img = img.resize((160, 160))
        img_array = np.array(test_img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        try:
            prediction = model.predict(img_array)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prediction = None

        if prediction is not None:
            prediction = np.asarray(prediction)

            if prediction.ndim == 2 and prediction.shape[-1] == 2:
                probs = prediction[0]
                class_idx = int(np.argmax(probs))
                score = float(probs[class_idx])
                label = "REAL" if class_idx == 1 else "FAKE"
                conf = score * 100
            elif prediction.ndim == 2 and prediction.shape[-1] == 1:
                score = float(prediction[0, 0])
                label = "REAL" if score >= 0.65 else "FAKE"
                conf = score * 100 if label == "REAL" else (1.0 - score) * 100
            else:
                flat = prediction.flatten()
                if flat.size == 1:
                    score = float(flat[0])
                    label = "REAL" if score >= 0.65 else "FAKE"
                    conf = score * 100 if label == "REAL" else (1.0 - score) * 100
                else:
                    class_idx = int(np.argmax(flat))
                    score = float(flat[class_idx])
                    label = "REAL" if class_idx == 1 else "FAKE"
                    conf = score * 100

            if label == "REAL":
                st.success("### Result: REAL")
            else:
                st.error("### Result: FAKE")

            st.metric("Confidence Score", f"{conf:.2f}%")

    # Metadata Section
    st.divider()
    st.subheader("📁 Digital Forensics (EXIF)")
    meta = get_metadata(uploaded_file)
    if meta:
        col1, col2 = st.columns(2)
        col1.write(f"**Manufacturer:** {meta['Make']}")
        col1.write(f"**Device Model:** {meta['Model']}")
        col2.write(f"**Software Used:** {meta['Software']}")
        col2.write(f"**Original Date:** {meta['Date']}")
    else:
        st.warning("No metadata found. Many AI generators and social media platforms strip this data.")
