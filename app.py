import cv2
import pytesseract
import imutils
import streamlit as st
import numpy as np
from PIL import Image

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="ANPR AI System",
    page_icon="🚘",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    .title {
        font-size: 50px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 20px;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 30px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
    }

    .result-box {
        background-color: #111827;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #334155;
        color: white;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown(
    '<div class="title">🚘 Automatic Number Plate Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered vehicle number plate detection using OpenCV and OCR</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# TESSERACT PATH
# -----------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("⚙️ About Project")

st.sidebar.info(
    """
    This ANPR system uses:

    • OpenCV for image processing
    • OCR for text recognition
    • Streamlit for web UI
    • Python for backend logic
    """
)

st.sidebar.success("Project Status: Working ✅")

# -----------------------------------
# FILE UPLOADER
# -----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# PROCESS IMAGE
# -----------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = imutils.resize(image, width=700)

    # Create columns
    col1, col2 = st.columns(2)

    # -----------------------------------
    # DISPLAY ORIGINAL IMAGE
    # -----------------------------------
    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    # -----------------------------------
    # IMAGE PROCESSING
    # -----------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 170, 200)

    # -----------------------------------
    # FIND CONTOURS
    # -----------------------------------
    cnts, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = sorted(
        cnts,
        key=cv2.contourArea,
        reverse=True
    )[:30]

    number_plate = None

    for c in cnts:

        perimeter = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(
            c,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:
            number_plate = approx
            break

    # -----------------------------------
    # DETECTION RESULT
    # -----------------------------------
    if number_plate is None:

        st.error("❌ No Number Plate Detected")

    else:

        cv2.drawContours(
            image,
            [number_plate],
            -1,
            (0, 255, 0),
            3
        )

        x, y, w, h = cv2.boundingRect(number_plate)

        cropped = gray[y:y+h, x:x+w]

        # OCR
        text = pytesseract.image_to_string(
            cropped,
            config='--psm 8'
        )

        text = text.strip()

        # -----------------------------------
        # DISPLAY RESULTS
        # -----------------------------------
        with col2:

            st.subheader("🎯 Detection Result")

            st.image(
                cropped,
                caption="Detected Number Plate",
                use_container_width=True
            )

            st.markdown(
                f'<div class="result-box">{text}</div>',
                unsafe_allow_html=True
            )

        st.subheader("✅ Final Processed Output")

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        st.success("Number Plate Detected Successfully!")
