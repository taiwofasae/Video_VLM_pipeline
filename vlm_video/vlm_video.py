import streamlit as st
import openai
import base64
import cv2
import os
import tempfile
from io import BytesIO
import pandas as pd
from fpdf import FPDF
import concurrent.futures

# Function to encode image as base64 for GPT-4 Turbo
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Function to check if GPU (CUDA) is available
def is_cuda_available():
    return cv2.cuda.getCudaEnabledDeviceCount() > 0

# Function to extract multiple frames from the video at specified intervals with GPU acceleration
def extract_frames(video_path, frame_interval=30, use_gpu=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []

    use_cuda = is_cuda_available() if use_gpu else False

    for frame_number in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if success:
            if use_cuda:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                frame = gpu_frame.download()  # Move back to CPU memory

            _, buffer = cv2.imencode(".jpg", frame)

            timestamp_sec = frame_number / fps  
            formatted_timestamp = f"{int(timestamp_sec // 3600):02}:{int((timestamp_sec % 3600) // 60):02}:{int(timestamp_sec % 60):02}"

            extracted_frames.append((frame_number, formatted_timestamp, buffer.tobytes()))

    cap.release()
    return extracted_frames

# Function to analyze frames in parallel using ThreadPoolExecutor
def analyze_frames_parallel(frames, prompt, api_key, base_model):
    def analyze_single_frame(frame_data):
        frame_number, timestamp_hms, frame_bytes = frame_data
        response = analyze_frame_with_gpt4_turbo(frame_bytes, prompt, api_key, base_model)
        return {
            "Frame": frame_number,
            "Timestamp": timestamp_hms,
            "Detected Actions": response
        }

    # Use ThreadPoolExecutor for parallel frame analysis
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(analyze_single_frame, frames))

    return results

# Function to analyze an extracted frame with GPT-4 Turbo for actions
def analyze_frame_with_gpt4_turbo(frame_bytes, prompt, api_key, base_model):
    client = openai.OpenAI(api_key=api_key)
    try:
        base64_image = encode_image(frame_bytes)

        response = client.chat.completions.create(
            model=base_model,
            messages=[
                {"role": "system", "content": "You are an AI assistant analyzing images and detecting actions happening in the video."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"{prompt}. Also, describe the actions happening in this frame."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=300
        )

        return response.choices[0].message.content  

    except openai.OpenAIError as e:
        return f"❌ OpenAI API Error: {str(e)}"

# Function to create a PDF with filtered results
def create_pdf(filtered_results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Filtered Video Analysis Results", ln=True, align='C')
    pdf.ln(10)

    for frame_data in filtered_results:
        frame_text = f"Frame {frame_data['Frame']} (Time: {frame_data['Timestamp']}): {frame_data['Detected Actions']}"
        frame_text = frame_text.encode("latin-1", "replace").decode("latin-1")

        pdf.multi_cell(0, 10, frame_text)
        pdf.ln(5)

    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)

    return pdf_buffer

# Streamlit App
def vlm_gbt(api_key, base_model):
    """Vision-Language Model UI with GPT-4 Turbo for Video Action Analysis"""

    st.title("📹 Vision-Language Model for Video Action Analysis")
    st.write("Upload a video and enter a prompt to analyze frames and detect actions.")

    uploaded_file = st.file_uploader("📤 Upload a video...", type=["mp4", "avi", "mov", "mkv"])
    prompt = st.text_input("💬 Enter a prompt", "Describe the video and actions happening in it")
    keyword = st.text_input("🔍 Enter keyword to filter results", "")
    frame_interval = st.number_input("🎞 Select frame extraction interval", min_value=1, value=30, step=10)
    use_gpu = st.checkbox("⚡ Enable GPU Acceleration (CUDA)", value=True)

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
            with st.spinner("🔎 Extracting frames and analyzing actions... (Parallel Processing Enabled)"):
                frames = extract_frames(video_path, frame_interval, use_gpu=use_gpu)
                filtered_results = analyze_frames_parallel(frames, prompt, api_key, base_model)

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

        if clear_memory_clicked:
            st.session_state.filtered_results = []
            st.rerun()
