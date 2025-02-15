# Vision-Language Model for Video Action Analysis 

This project is a **Streamlit-based AI Assistant** that allows users to **analyze video frames, detect actions, and extract meaningful insights** using OpenAI's GPT-4 Turbo.

---

## 📌 Features
- **Parallel Processing**: Uses `ThreadPoolExecutor` to analyze multiple frames concurrently.
- **GPU Acceleration**: Leverages OpenCV’s `cuda` backend for faster frame decoding (if supported).
- **Frame Extraction**: Extracts frames from a video at a user-defined interval.
- **AI-Powered Action Detection**: Uses GPT-4 Turbo to analyze and describe actions in frames.
- **Keyword-Based Filtering**: Filters frames based on user-defined keywords.
- **Timestamped Results**: Each frame includes a **timestamp** (HH:MM:SS format) for accuracy.
- **Downloadable Reports**: Save results as **CSV** and **PDF** for later use.
- **Secure API Access**: Requires an **OpenAI API Key** to ensure authorized usage.

---

## 🏗️ Project Structure
```
📁 video-action-analysis
│── 📄 main.py                # Streamlit app entry point
│── 📁 vlm_video
│   │── 📄 vlm_video.py       # Video processing & analysis module
│── 📁 utils
│   │── 📄 utils.py           # Utility functions
│── 📄 requirements.txt       # Dependencies
│── 📄 README.md              # Project documentation
```

---

## Installation
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/TimGoebel/Video_VLM_pipeline.git
cd video-action-analysis
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## Running the App
Run the Streamlit app with:
```sh
streamlit run main.py
```

or use **run.bat** (for Windows users):
```
@echo off
python -m streamlit run main.py
pause  # Remove this if you don't want the CMD screen to stay open for troubleshooting
```

---

## How to Use

1️⃣ **Run the app** using Streamlit.  
2️⃣ **Enter your OpenAI API key** in the sidebar.  
3️⃣ **Select GPT-4 Turbo** as the model.  
4️⃣ **Upload a video** file.  
5️⃣ **Set frame extraction interval.**  
6️⃣ **Enable GPU acceleration (if supported).**  
7️⃣ **Enter a keyword** to filter relevant actions.  
8️⃣ **Analyze video** and view detected actions.  
9️⃣ **Download results** in CSV, PDF, or annotated video format.  
🔟 **Clear all stored session data only when explicitly requested.**  

---

## Performance Optimization
### **Parallel Frame Processing**
- Extracts frames concurrently using `ThreadPoolExecutor`.
- Reduces analysis time by parallelizing GPT-4 Turbo calls.

### **GPU Acceleration with OpenCV CUDA**
- Uses CUDA (`cv2.cuda_GpuMat`) for **faster frame decoding**.
- Falls back to CPU if CUDA is unavailable.

---

##  Future Enhancements
**Object Detection** 
  - Adding YOLOor some other type bounding box detection.
    
**Model Enhancements**
  - Multi-Frame Context Analysis: Instead of analyzing frames independently, use consecutive frames to infer temporal actions (e.g., using an LSTM or Transformer model).
  - Action Categorization & Confidence Scores: Fine-tune response generation to provide action category labels and confidence scores.

**Additional UX Improvements**
   - Multilingual Support: Allow users to select a language for analysis.
   - Speech-to-Text for Audio Analysis: Extract insights from spoken words in the video.
   - Auto-Summarization: Provide a summary of detected actions instead of just listing them.
---

## 📜 License
This project is licensed under the **MIT License**.

---



