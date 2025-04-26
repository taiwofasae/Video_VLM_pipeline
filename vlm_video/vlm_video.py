import streamlit as st
import openai
import base64
import cv2
import os
import tempfile
import numpy as np
from io import BytesIO
import pandas as pd
from fpdf import FPDF
import concurrent.futures
import re

# Function to encode image as base64 for GPT-4 Turbo
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Function to check if GPU (CUDA) is available
def is_cuda_available():
    return cv2.cuda.getCudaEnabledDeviceCount() > 0

# Function to extract multiple frames with multi-frame context
def extract_multi_frame_context(video_path, frame_interval=30, sequence_length=5, use_gpu=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_sequences = []
    frame_buffer = []

    for frame_number in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        
        if success:
            _, buffer = cv2.imencode(".jpg", frame)
            timestamp_sec = frame_number / fps
            formatted_timestamp = f"{int(timestamp_sec // 3600):02}:{int((timestamp_sec % 3600) // 60):02}:{int(timestamp_sec % 60):02}"
            frame_buffer.append((frame_number, formatted_timestamp, buffer.tobytes()))
            
            if len(frame_buffer) == sequence_length:
                extracted_sequences.append(frame_buffer.copy())
                frame_buffer.pop(0)
    
    cap.release()
    return extracted_sequences

# Function to create a PDF with filtered results
def create_pdf(filtered_results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Filtered Video Analysis Results", ln=True, align='C')
    pdf.ln(10)

    for frame_data in filtered_results:
        frame_text = f"Frame Start: {frame_data['Frame Start']} - End: {frame_data['Frame End']} - Actions: {frame_data['Detected Actions']} - Objects: {frame_data['Detected Objects']}"
        frame_text = frame_text.encode("latin-1", "replace").decode("latin-1")

        pdf.multi_cell(0, 10, frame_text)
        pdf.ln(5)

    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)

    return pdf_buffer

# Function to analyze sequences of frames with GPT-4 Turbo
def analyze_frame_sequence(sequence, prompt, search_object, api_key, base_model):
    client = openai.OpenAI(api_key=api_key)
    try:
        image_payload = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame[2])}"}}
            for frame in sequence
        ]
        
        response = client.chat.completions.create(
            model=base_model,
            messages=[
                {"role": "system", "content": "You are an AI assistant analyzing sequences of images to detect actions and their confidence scores."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"""{prompt}. Two things:
                    1.  Analyze actions across these frames and provide action categories with confidence scores.

                    2.  More importantly, find objects in the video frames and provide in this list format:
                        ---start list---
                        Object1
                        Object2
                        ---end list---

                    The frames are attached.
                    """}
                ] + image_payload}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content  
    
    except openai.OpenAIError as e:
        return f"❌ OpenAI API Error: {str(e)}"

def extract_objects_from_response(response):
    print(f'Response: {response}')
    # 1. Grab everything between the markers
    m = re.search(r"---start list---\s*(.*?)\s*---end list---", response, re.DOTALL)
    if not m:
        return []
        pass#raise ValueError("Couldn't find the start/end markers")

    block = m.group(1)

    # 2. Split into lines, strip out any blank ones
    objects = [line.strip() for line in block.splitlines() if line.strip()]

    return objects

# Function to analyze multi-frame sequences in parallel
def analyze_sequences_parallel(sequences, prompt, search_object, api_key, base_model):
    def analyze_single_sequence(sequence):
        response = analyze_frame_sequence(sequence, prompt, search_object, api_key, base_model)
        return {
            "Frame Start": sequence[0][1],  
            "Frame End": sequence[-1][1],  
            "Detected Actions": response,
            "Detected Objects": ', '.join(extract_objects_from_response(response))
        }
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(analyze_single_sequence, sequences))
    
    return results

# Streamlit App
def vlm_gbt(api_key, base_model):
    st.title("📹 Vision-Language Model for Multi-Frame Video Action Analysis")
    st.write("Upload a video to analyze sequences of frames and detect actions with confidence scores.")
    
    uploaded_file = st.file_uploader("📤 Upload a video...", type=["mp4", "avi", "mov", "mkv"])
    prompt = st.text_input("💬 Enter a prompt", "Describe the video and actions happening in it")
    frame_interval = st.number_input("🎞 Select frame extraction interval", min_value=1, value=30, step=10)
    sequence_length = st.number_input("🔗 Number of Frames per Sequence", min_value=2, value=5, step=1)
    keyword = st.text_input("🔍 Enter keyword to filter results", "")

    search_object = st.text_input(" Enter object to find in frames", "")
    
    if "filtered_results" not in st.session_state:
        st.session_state.filtered_results = []

    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.video(video_path)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            analyze_clicked = st.button("🔍 Analyze & Filter by Keyword")
        
        with col2:
            clear_memory_clicked = st.button("🗑 Clear ChatGPT Memory")
        
        if analyze_clicked:
            with st.spinner("🔎 Extracting frames and analyzing actions..."):
                sequences = extract_multi_frame_context(video_path, frame_interval, sequence_length)
                filtered_results = analyze_sequences_parallel(sequences, prompt, search_object, api_key, base_model)
                
                if keyword:
                    filtered_results = [res for res in filtered_results if keyword.lower() in res["Detected Actions"].lower()]
                
                if filtered_results:
                    st.session_state.filtered_results = filtered_results
                else:
                    st.error("❌ No relevant results found.")

        if st.session_state.filtered_results:
            results_df = pd.DataFrame(st.session_state.filtered_results)
            st.subheader(f"📖 Filtered Results for Keyword: '{keyword}'")
            st.dataframe(results_df)

            csv_buffer = BytesIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            pdf_buffer = create_pdf(st.session_state.filtered_results)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button("📥 Download CSV", data=csv_buffer, file_name="filtered_video_analysis.csv", mime="text/csv")

            with col2:
                st.download_button("📥 Download PDF", data=pdf_buffer, file_name="filtered_video_summary.pdf", mime="application/pdf")

        #**Clear all stored session data only when explicitly requested**
        if clear_memory_clicked:
            st.session_state.filtered_results = []
            st.rerun()
