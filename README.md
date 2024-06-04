# 🎥 Interview Analysis App 🤖

Analyze interview videos to assess candidate suitability using cutting-edge AI technology! This app extracts frames from videos, detects emotions and postures, generates comprehensive summaries, and visualizes results.

## Features

- 🎞️ **Extract Frames**: Extract frames from the video at specified intervals.
- 😃 **Detect Emotions**: Analyze frames for emotions (Happy, Fear, Surprise).
- 🕺 **Analyze Postures**: Detect postures using advanced pose detection.
- 📝 **Generate Summaries**: Create detailed summaries using OpenAI's GPT-3.5-turbo.
- 📊 **Visualize Results**: Display emotion scores over time and average scores in charts.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-username/interview-analysis-app.git
   cd interview-analysis-app

2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt

3. **Set Up OpenAI API Key:**:
   ```sh
   export OPENAI_API_KEY='your-api-key-here'

### Running the App on Local Device

1. **Start the Streamlit App:**
   ```sh
   streamlit run app.py

2. **Upload a Video:**
   - Use the sidebar to upload a video file.
   - Set the frame extraction interval (in minutes).
  
3. **Analyze the Video:**
   - The app will extract frames, detect emotions and postures, and generate a summary.
   - View emotion charts and save the analysis to a CSV file.

### Usage

1. **Extract Frames**
   Frames are extracted from the video at specified intervals and saved as images.

2. **Detect Emotions**
   Emotions (Happy, Fear, Surprise) are detected in each frame using a pre-trained model.

3. **Detect Postures**
   Postures are analyzed using MediaPipe's pose detection.

4. **Generate Summary**
   A comprehensive summary of the candidate's performance is generated using OpenAI's GPT-3.5-turbo.

5. **Visualize Results**
   Emotion scores over frames and average scores are displayed in line and bar charts.

### Contributing
  Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

### Contact
  For questions or support, please contact:
  * Email: abdullahhaneef08@gmail.com
  * GitHub: https://github.com/abdullah-haneef
  * LinkedIn: https://linkedin.com/in/abdullah-haneef
