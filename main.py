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
    first_frame_path = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
            if first_frame_path is None:
                first_frame_path = frame_filename
        count += 1

    cap.release()
    return first_frame_path


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


def create_emotion_chart(emotion_results):
    data = []
    important_emotions = ['happy', 'fear', 'surprise']

    for frame, emotions in emotion_results.items():
        for emotion in emotions:
            if emotion['label'].lower() in important_emotions:
                data.append({
                    'Frame': frame,
                    'Label': emotion['label'],
                    'Score': emotion['score']
                })

    df = pd.DataFrame(data)
    df['Frame'] = pd.to_numeric(df['Frame'].str.replace('frame_', '').str.replace('.jpg', ''))
    df = df.sort_values(by='Frame')

    plt.figure(figsize=(14, 8))
    for label in df['Label'].unique():
        subset = df[df['Label'] == label]
        plt.plot(subset['Frame'], subset['Score'], label=label)

    plt.xlabel('Frames')
    plt.ylabel('Emotion Score')
    plt.title('Emotion Scores Over Frames')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def create_average_emotion_chart(emotion_results):
    data = []
    important_emotions = ['happy', 'fear', 'surprise']

    for frame, emotions in emotion_results.items():
        for emotion in emotions:
            if emotion['label'].lower() in important_emotions:
                data.append({
                    'Label': emotion['label'].lower(),
                    'Score': emotion['score']
                })

    df = pd.DataFrame(data)

    # Group by 'Label' and calculate the mean score for each emotion
    df_mean = df.groupby('Label').mean().reset_index()

    # Plotting the average scores
    plt.figure(figsize=(10, 6))
    plt.bar(df_mean['Label'], df_mean['Score'], color=['red', 'green', 'orange'])
    plt.xlabel('Emotion')
    plt.ylabel('Average Score')
    plt.title('Average Scores of Emotions')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)


# OpenAI API Key
openai.api_key = 'sk-proj-zYWjQyWEDOoFfDVNf3szT3BlbkFJIEV7tw7CcoxSq35cOtr4'

# Function to generate a comprehensive summary
def generate_summary(emotion_results, posture_results):
    prompt = '''
    Based on the following observations from the interview, assess the suitability of the candidate.
    Consider emotions and postures detected.

    Overall Suitability Score: Calculated Score out of 10 by assigning scores to emotions (happy:2, surprise:0, fear:-2)
    Overall Result: Suitable if candidate score is greater than 6,  Not suitable if candidate score is less than or equal to 6
    Reasoning: Explain how specific emotions and postures influenced the candidate's performance. Discuss strengths and areas for improvement.

    Generate output in the following format and do not use frame numbers in the analysis at all:

    Based on the observations from the interview, here is the assessment of the candidate's suitability:

    Candidate Assessment:
    
    Overall Suitability Score: X/10

    Overall Result:

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

    return df


# Streamlit app
st.set_page_config(page_title='Interview Analysis', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

# Apply custom CSS for black background and white text
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Handle page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'main'

def main_page():
    st.title('🎥 Interview Analysis')
    
    time_interval = st.slider('Frame Extraction Interval (minutes)', 1, 5, 1)
    
    st.header('Upload Video')
    video_file = st.file_uploader('Choose a video file', type=['mp4', 'avi', 'mov', 'mkv'])
    
    if video_file is not None:
        with st.spinner('Extracting frames...'):
            with open('temp_video.mp4', 'wb') as f:
                f.write(video_file.read())
            first_frame_path = extract_frames('temp_video.mp4', time_interval=time_interval)
    
        if first_frame_path:
            st.image(first_frame_path, caption='First extracted frame from the video', use_column_width=True)
    
        with st.spinner('Detecting emotions...'):
            emotion_results = detect_emotions()
    
        with st.spinner('Detecting postures...'):
            posture_results = detect_postures()
    
        with st.spinner('Generating summary...'):
            analysis_output = generate_summary(emotion_results, posture_results)
    
        st.write(analysis_output)
    
        with st.spinner('Creating emotion charts...'):
            create_emotion_chart(emotion_results)
            create_average_emotion_chart(emotion_results)
    
        # Get video title from user
        video_title = st.text_input('Enter the video title', '')
    
        if st.button('Save Analysis'):
            df = save_analysis_to_csv(video_title, analysis_output)
            st.success('Analysis saved to CSV file.')
    
            # Provide download link for the CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='analysis_results.csv',
                mime='text/csv',
            )


# "Behind the Scenes" page
def behind_the_scenes_page():
    st.title('Behind the Scenes')
    st.write(
        """
        
        ### Behind the Scenes
        Here's how the code is composed of different functions and their description:
        
        1. extract_frames:
        - Extracts frames from a video file at specified time intervals.
        - Inputs:
            - video_path: Path to the video file.
            - output_folder: Folder to save the extracted frames (default is 'frames').
            - time_interval: Interval in minutes between frame extractions (default is 1).
        - Returns the path of the first extracted frame.
        
        2. detect_emotions:
        - Detects emotions in images stored in a specified folder.
        - Inputs:
            - frame_folder: Path to the folder containing frames (default is 'frames').
        - Returns a dictionary of emotion analysis results.

        3. detect_postures:
        - Analyzes postures in images stored in a specified folder.
        - Inputs:
            - frame_folder: Path to the folder containing frames (default is 'frames').
        - Returns a dictionary of posture analysis results.

        4. create_emotion_chart:
        - Generates a line chart of emotion scores over video frames.
        - Inputs:
            - emotion_results: Dictionary of emotion analysis results.

        5. create_average_emotion_chart:
        - Generates a bar chart showing the average scores of specific emotions across all frames.
        - Inputs:
            - emotion_results: Dictionary of emotion analysis results.

        6. generate_summary:
        - Generates a comprehensive summary of the candidate's performance based on detected emotions and postures.
        - Reads a prompt template from a configuration file.
        - Uses OpenAI's GPT-3.5-turbo model to generate the summary.
        - Returns the generated summary text.

        7. save_analysis_to_csv:
        - Saves the analysis results to a CSV file.
        - Inputs:
            - video_title: Title of the analyzed video.
            - analysis_output: Generated analysis summary.
            - output_file: Name of the output CSV file (default is 'analysis_results.csv').
        - Creates a DataFrame with the video title and analysis output.
        - Appends the data to the specified CSV file or creates a new file if it does not exist.
        - Returns the DataFrame containing the analysis data.
        
        """
    )

st.sidebar.header('Navigation')
if st.sidebar.button('Main Page'):
    st.session_state.page = 'main'

if st.sidebar.button('Behind the Scenes'):
    st.session_state.page = 'behind_the_scenes'

# Additional Information
st.sidebar.header('📋 Project Description')
st.sidebar.write(
    """
    🎥 Interview Analysis App 🤖
    

    🔍 Analyze Interview Videos to Assess Candidate Suitability! 🧑‍💼👩‍💼
    

    ✨ Features:
    

    🎞️ Extract Frames from the Video
    
    😃 Detect Emotions (Happy, Fear, Surprise) in Each Frame
    
    🕺 Analyze Postures with Advanced Pose Detection
    
    📝 Generate Comprehensive Summaries Using OpenAI's GPT-3.5-turbo
    
    🚀 Transform Your Interview Process with Cutting-Edge AI Technology!


    """
)

st.sidebar.header('Contact')
st.sidebar.write(
    """
    
    📬 Contact Information 🌐
    

    📧 Email: [abdullahhaneef08@gmail.com](mailto:abdullahhaneef08@gmail.com)
    
    🐱‍💻 GitHub: [https://github.com/abdullah-haneef]
    
    💼 LinkedIn: [https://www.linkedin.com/in/abdullah-haneef/]
    
    """
)

# Page navigation
if st.session_state.page == 'behind_the_scenes':
    behind_the_scenes_page()
else:
    main_page()
