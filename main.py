import os
import cv2
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from transformers import pipeline
import mediapipe as mp

# Initialize emotion detection model
emotion_model = pipeline('image-classification', model='dima806/facial_emotions_image_detection')

# Initialize posture detection model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


# Function to extract frames from video
def extract_frames(video_path, output_folder='frames', time_interval=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * 60 * time_interval)  # Convert minutes to frames

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()


# Function to detect emotions in frames
def detect_emotions(frame_folder='frames'):
    emotion_results = {}
    for frame_filename in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame_filename)
        try:
            image = Image.open(frame_path)
            result = emotion_model(image)
            emotion_results[frame_filename] = result
        except Exception as e:
            st.error(f"Error processing image {frame_filename}: {e}")
    return emotion_results


# Function to detect postures in frames
def detect_postures(frame_folder='frames'):
    posture_results = {}
    for frame_filename in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame_filename)
        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)
        posture_results[frame_filename] = result
    return posture_results


# Function to create emotion chart
def create_emotion_chart(emotion_results):
    data = []
    important_emotions = ['happy', 'neutral', 'sad']

    for frame, emotions in emotion_results.items():
        for emotion in emotions:
            if emotion['label'].lower() in important_emotions:
                data.append({
                    'Frame': frame,
                    'Label': emotion['label'],
                    'Score': emotion['score']
                })

    df = pd.DataFrame(data)
    df_pivot = df.pivot(index='Frame', columns='Label', values='Score').fillna(0)

    plt.figure(figsize=(14, 8))
    for label in df_pivot.columns:
        plt.plot(df_pivot.index, df_pivot[label], label=label)

    plt.xlabel('Frame')
    plt.ylabel('Score')
    plt.title('Detected Human Emotions Over Frames')
    plt.legend(loc='upper right')
    plt.xticks([])
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)


# OpenAI API Key
openai.api_key = 'sk-proj-dOHrboyQvaNpRvnvAfo5T3BlbkFJ3C4VmjNGKnKiUO8AOwbi'

# Function to generate a comprehensive summary
def generate_summary(emotion_results, posture_results):
    prompt = '''
    Based on the following observations from the interview, assess the suitability of the candidate.
    Consider emotions and postures detected.

    Overall Result: Suitable / Not suitable
    Overall Suitability Score: Calculated Score out of 10 by assigning scores to emotions (happy:2, neutral:0, sad:-2)
    Reasoning: Explain how specific emotions and postures influenced the candidate's performance. Discuss strengths and areas for improvement.

    Generate output in the following format:

    Based on the observations from the interview, here is the assessment of the candidate's suitability:

    Candidate Assessment:
    Overall Result:
    Overall Suitability Score: X/10

    Reasoning:

    Strengths:

    Areas for Improvement:

    '''

    frame_count = min(5, len(emotion_results))
    for frame_filename in sorted(emotion_results.keys())[:frame_count]:
        emotions = emotion_results[frame_filename]
        posture = posture_results[frame_filename]
        prompt += f"Frame {frame_filename}: Emotion - {emotions}, Pose - {posture.pose_landmarks}\n"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=500,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    return response['choices'][0]['message']['content']

# Function to save analysis to CSV
def save_analysis_to_csv(video_title, analysis_output, output_file='analysis_results.csv'):
    data = {
        'Video Title': [video_title],
        'Analysis': [analysis_output]
    }

    df = pd.DataFrame(data)
    
    if not os.path.isfile(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)

# Streamlit app
st.set_page_config(page_title='Interview Analysis', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

st.markdown(
    """
    <style>
    .reportview-container {
        background: #000;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Interview Analysis')

# Sidebar
st.sidebar.header('Project Description')
st.sidebar.write(
    """
    This app analyzes interview videos to assess the suitability of candidates. 
    It extracts frames from the video, detects emotions and postures, and generates 
    a comprehensive summary using OpenAI's GPT-3.5-turbo.
    """
)

st.sidebar.header('Contact')
st.sidebar.write(
    """
    - **Email:** abdullahhaneef08@gmail.com
    - **GitHub:** https://github.com/abdullah-haneef
    - **LinkedIn:** https://www.linkedin.com/in/abdullah-haneef/
    """
)

st.sidebar.header('Upload Video')
video_file = st.sidebar.file_uploader('Choose a video file', type=['mp4', 'avi', 'mov', 'mkv'])

time_interval = st.sidebar.slider('Frame Extraction Interval (minutes)', 1, 10, 1)

if video_file is not None:
    st.sidebar.write('Extracting frames...')
    with open('temp_video.mp4', 'wb') as f:
        f.write(video_file.read())
    extract_frames('temp_video.mp4', time_interval=time_interval)

    st.sidebar.write('Detecting emotions...')
    emotion_results = detect_emotions()

    st.sidebar.write('Detecting postures...')
    posture_results = detect_postures()

    st.sidebar.write('Creating emotion chart...')
    create_emotion_chart(emotion_results)

    st.sidebar.write('Generating summary...')
    analysis_output = generate_summary(emotion_results, posture_results)

    st.write(analysis_output)

    # Get video title from user
    video_title = st.text_input('Enter the video title', '')

    if st.button('Save Analysis'):
        save_analysis_to_csv(video_title, analysis_output)
        st.success('Analysis saved to CSV file.')
