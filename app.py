import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
import requests
from pathlib import Path

# Fix for Windows path compatibility
pathlib.PosixPath = pathlib.WindowsPath

# Function to download the model file from GitHub
def download_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Create a temporary file to store the model
        temp_model = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        temp_model.write(response.content)
        temp_model.close()
        return temp_model.name
    else:
        raise Exception(f"Failed to download the model. Status code: {response.status_code}")

# Set your model and repository paths
repo_path = 'Giri220201/yolov5'  # GitHub repository for YOLOv5
model_url = 'https://github.com/Giri220201/yolov5/raw/refs/heads/master/tennis.pt'

st.title("🎾 Tennis Game Tracking")

# Download and load the YOLOv5 model
try:
    st.info("Downloading model...")
    model_path = download_model(model_url)  # Download the model file from GitHub
    model = torch.hub.load(repo_path, 'custom', path=model_path, source='github')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Custom styles for buttons and progress bar
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 10px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    .stProgress > div > div {
        background-color: #4CAF50; /* Green progress bar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create two columns for layout
col1, col2 = st.columns([3, 1])

# Video upload in the first column
with col1:
    st.subheader("Upload & Preview Video")
    video_file = st.file_uploader("Select Input File", type=["mp4", "avi", "mov"])
    if video_file is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(video_file.read())

        if st.button("Preview Video"):
            st.video(temp_video.name)
        else:
            st.write("Click 'Preview Video' to view the uploaded video.")
    else:
        st.write("Please select a video file.")

# Download button in the second column
with col2:
    st.subheader("Download Processed Video")
    if video_file is not None and st.button("Download Output"):
        st.write("Processing and preparing download...")

        # Process the video with YOLOv5 detection
        cap = cv2.VideoCapture(temp_video.name)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = model(frame)
            frame = np.squeeze(results.render())  # Draw the detection boxes on the frame

            # Write the frame to the output video
            out.write(frame)

        cap.release()
        out.release()

        # Provide a download link for the processed video
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

st.write("Ensure the YOLOv5 repository is accessible and properly formatted.")
