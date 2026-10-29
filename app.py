import os
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

import link_detector
import sms_detector
import deepfake_detector

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Deepfake and Fraud Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# Model Loading
# ==============================================================================
@st.cache_resource
def load_models():
    df_model, df_device = deepfake_detector.load_or_init_model()
    sms_model = sms_detector.get_model()
    return df_model, df_device, sms_model

df_model, df_device, sms_model = load_models()

# ==============================================================================
# Main Page Header
# ==============================================================================
st.title("Deepfake and Fraud Detection System")
st.caption("2nd Year 3rd Sem APP Project")
st.write("")

# 3 Separate Systems
tab_deepfake, tab_sms, tab_link = st.tabs([
    "Deepfake Detection",
    "SMS Scam Detection",
    "Link Detection"
])

# ==============================================================================
# 1. DEEPFAKE DETECTION (IMAGE & VIDEO)
# ==============================================================================
with tab_deepfake:
    st.subheader("Deepfake Detection")
    st.write("Upload an image or video to check whether the face has been manipulated.")

    media_type = st.radio("Input format:", ["Image", "Video"], horizontal=True)

    if media_type == "Image":
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_image is not None:
            image_pil = Image.open(uploaded_image).convert("RGB")
            img_np = np.array(image_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with col1:
                st.image(image_pil, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Analyzing image..."):
                face_crop, face_found, bbox = deepfake_detector.crop_face(img_bgr)
                result = deepfake_detector.analyze_image(img_bgr, df_model, df_device)

            with col2:
                st.markdown("### Analysis Result")
                fake_pct = result["fake_confidence"]
                real_pct = result["real_confidence"]

                if fake_pct >= 50.0:
                    st.error(f"This image is {fake_pct}% likely to be a deepfake.")
                else:
                    st.success(f"This image is {real_pct}% likely to be authentic ({fake_pct}% deepfake risk).")

                st.progress(fake_pct / 100.0)
                st.metric("Deepfake Probability", f"{fake_pct}%")

                st.write("---")
                if face_found:
                    st.write("Detected Face Crop:")
                    st.image(face_crop, width=160)
                else:
                    st.info("No face detected by OpenCV; analyzed center crop.")

    else:
        # Video mode
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_video = st.file_uploader("Choose a video file:", type=["mp4", "avi", "mov", "mkv"])
            frame_interval = st.slider("Frame sampling interval:", min_value=5, max_value=30, value=15, step=5,
                                       help="Samples every Nth frame to reduce compute time.")

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            with col1:
                st.video(video_path)

            if st.button("Run Video Analysis"):
                with st.spinner("Analyzing video frames..."):
                    res = deepfake_detector.analyze_video(video_path, df_model, df_device, frame_interval=frame_interval)

                with col2:
                    st.markdown("### Video Analysis Result")
                    avg_fake = res['avg_fake_score']
                    peak_fake = res['peak_fake_score']

                    if avg_fake >= 50.0:
                        st.error(f"This video is {avg_fake}% likely to be a deepfake (Peak: {peak_fake}%).")
                    else:
                        st.success(f"This video is {round(100.0 - avg_fake, 1)}% likely to be authentic ({avg_fake}% deepfake risk).")

                    st.progress(avg_fake / 100.0)
                    c1, c2 = st.columns(2)
                    c1.metric("Average Deepfake Probability", f"{avg_fake}%")
                    c2.metric("Peak Frame Probability", f"{peak_fake}%")

                    c3, c4 = st.columns(2)
                    c3.metric("Frames Analyzed", res["frames_analyzed"])
                    c4.metric("Flagged Frames", f"{res['fake_frames_flagged']} ({res['fake_frame_percentage']}%)")

                if "frame_details" in res and len(res["frame_details"]) > 0:
                    st.write("---")
                    st.write("Frame-by-frame Fake Score:")
                    chart_df = pd.DataFrame({
                        "Timestamp (s)": [f["timestamp"] for f in res["frame_details"]],
                        "Fake Score (%)": [f["fake_score"] for f in res["frame_details"]]
                    })
                    st.line_chart(chart_df.set_index("Timestamp (s)"))

            try:
                os.remove(video_path)
            except Exception:
                pass


# ==============================================================================
# 2. SMS SCAM DETECTION
# ==============================================================================
with tab_sms:
    st.subheader("SMS Scam Detection")
    st.write("Enter an SMS message to classify whether it is a scam or a legitimate message.")

    sms_input = st.text_area("Message text:", height=120, placeholder="Type or paste the message here...")

    if st.button("Check SMS"):
        if not sms_input.strip():
            st.warning("Please enter a message to analyze.")
        else:
            is_scam, confidence = sms_detector.check_sms(sms_input, sms_model)

            st.write("---")
            if is_scam:
                st.error(f"This message is {confidence}% likely to be a scam.")
                st.progress(confidence / 100.0)
                st.metric("Scam Probability", f"{confidence}%")
            else:
                safe_conf = round(100.0 - confidence, 1)
                st.success(f"This message is {safe_conf}% likely to be legitimate ({confidence}% scam risk).")
                st.progress(safe_conf / 100.0)
                st.metric("Legitimate Probability", f"{safe_conf}%")

            # Check if message contains URLs
            links = link_detector.extract_links_from_text(sms_input)
            if links:
                st.write("Found Links in Message:")
                for lk in links:
                    st.write(f"- `{lk}`")
                    analysis = link_detector.analyze_link(lk)
                    st.caption(f"Status: {analysis['status']} (Suspicion Score: {analysis['score']})")


# ==============================================================================
# 3. LINK DETECTION
# ==============================================================================
with tab_link:
    st.subheader("Link Detection")
    st.write("Enter a website URL to check for phishing indicators, domain age, and security flags.")

    url_input = st.text_input("Website URL:", placeholder="e.g. http://example-site.com")

    if st.button("Check Link"):
        if not url_input.strip():
            st.warning("Please enter a URL to check.")
        else:
            with st.spinner("Checking domain and webpage headers..."):
                res = link_detector.analyze_link(url_input)

            st.write("---")
            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                scam_pct = res.get("scam_probability", 5.0)
                safe_pct = res.get("safe_probability", 95.0)

                if scam_pct >= 50.0:
                    st.error(f"This link is {scam_pct}% likely to be a scam.")
                else:
                    st.success(f"This link is {safe_pct}% likely to be safe ({scam_pct}% scam risk).")

                st.progress(scam_pct / 100.0)
                st.metric("Scam Probability", f"{scam_pct}%")

            with col_res2:
                st.write("Findings:")
                if res["reasons"]:
                    for r in res["reasons"]:
                        st.write(f"- {r}")
                else:
                    st.write("No suspicious flags detected.")
