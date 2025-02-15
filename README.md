📹 Vision-Language Model for Video Action Analysis

🚀 Overview

This project is a Streamlit-based application that leverages GPT-4 Turbo to analyze video frames and detect actions based on user-defined keywords. Users can upload a video, extract frames at a set interval, analyze them for actions, and download results in CSV and PDF formats.

📌 Features

✅ Frame Extraction: Extracts frames from a video at a user-defined interval.
✅ AI-Powered Action Detection: Uses GPT-4 Turbo to analyze and describe actions in frames.
✅ Keyword-Based Filtering: Filters frames based on user-defined keywords.
✅ Timestamped Results: Each frame includes a timestamp (HH:MM:SS format) for accuracy.
✅ Downloadable Reports: Save results as CSV and PDF for later use.
✅ Secure API Access: Requires an OpenAI API Key to ensure authorized usage.

🛠 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-repo/video-action-analysis.git
cd video-action-analysis

2️⃣ Install Dependencies

pip install -r requirements.txt

🚀 Running the App

Run the Streamlit app with:

streamlit run main.py

or use run.bat (for Windows users):

@echo off
python -m streamlit run main.py
pause  # Remove this if you don't want the CMD screen to stay open for troubleshooting

📂 Project Structure

📁 video-action-analysis
│── 📄 main.py                # Streamlit app entry point
│── 📁 vlm_video
│   │── 📄 vlm_video.py       # Video processing & analysis module
│── 📁 utils
│   │── 📄 utils.py           # Utility functions
│── 📄 requirements.txt       # Dependencies
│── 📄 README.md              # Project documentation

🎥 How to Use

1️⃣ Run the app using Streamlit.
2️⃣ Enter your OpenAI API key in the sidebar.
3️⃣ Select GPT-4 Turbo as the model.
4️⃣ Upload a video file.
5️⃣ Set frame extraction interval.
6️⃣ Enter a keyword to filter relevant actions.
7️⃣ Analyze video and view detected actions.
8️⃣ Download results in CSV or PDF format.
9️⃣ Clear all stored session data only when explicitly requested.

📜 License

This project is licensed under the MIT License.

