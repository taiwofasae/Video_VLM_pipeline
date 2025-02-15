import streamlit as st
import openai
import base64
import cv2
import os
import tempfile
from io import BytesIO
import pandas as pd
from fpdf import FPDF

# Function to encode image as base64 for GPT-4 Turbo
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Function to extract multiple frames from the video at specified intervals
def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []

    for frame_number in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if success:
            _, buffer = cv2.imencode(".jpg", frame)
            
            # Calculate timestamp in seconds
            timestamp_sec = frame_number / fps  
            
            # Convert timestamp to HH:MM:SS format
            hours = int(timestamp_sec // 3600)
            minutes = int((timestamp_sec % 3600) // 60)
            seconds = int(timestamp_sec % 60)
            formatted_timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"  # HH:MM:SS format
            
            extracted_frames.append((frame_number, formatted_timestamp, buffer.tobytes()))

    cap.release()
    return extracted_frames

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

    # Use a font that supports more characters
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Filtered Video Analysis Results", ln=True, align='C')
    pdf.ln(10)

    for frame_data in filtered_results:
        # Convert text to Latin-1 compatible encoding
        frame_text = f"Frame {frame_data['Frame']} (Time: {frame_data['Timestamp']}): {frame_data['Detected Actions']}"

        # Replace unsupported characters safely
        frame_text = frame_text.encode("latin-1", "replace").decode("latin-1")

        pdf.multi_cell(0, 10, frame_text)
        pdf.ln(5)

    # Create a BytesIO buffer for the PDF
    pdf_buffer = BytesIO()
    
    # Generate PDF output and write to BytesIO buffer
    pdf_output = pdf.output(dest='S').encode('latin1')  # Ensure Latin-1 encoding
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)  # Reset the buffer position for reading

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

    # Check if session state exists for previous results
    if "filtered_results" not in st.session_state:
        st.session_state.filtered_results = []

    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(video_path)

        # Create two columns for the buttons
        col1, col2 = st.columns([1, 1])

        with col1:
            analyze_clicked = st.button("🔍 Analyze & Filter by Keyword")

        with col2:
            clear_memory_clicked = st.button("🗑 Clear ChatGPT Memory")

        if analyze_clicked:
            with st.spinner("🔎 Extracting frames and analyzing actions..."):
                frames = extract_frames(video_path, frame_interval)
                filtered_results = []

                if frames:
                    for frame_number, timestamp_hms, frame_bytes in frames:
                        response = analyze_frame_with_gpt4_turbo(frame_bytes, prompt, api_key, base_model)
                        
                        # If the keyword is found in the response, store the result
                        if keyword.lower() in response.lower():
                            filtered_results.append({
                                "Frame": frame_number,
                                "Timestamp": timestamp_hms,  # Video timestamp in HH:MM:SS format
                                "Detected Actions": response
                            })

                    if filtered_results:
                        # Save results to session state to persist them
                        st.session_state.filtered_results = filtered_results
                else:
                    st.error("❌ Unable to extract frames from the video.")

        # **Display Results and Provide Downloads (Without Clearing Data)**
        if st.session_state.filtered_results:
            # Convert filtered results to DataFrame
            results_df = pd.DataFrame(st.session_state.filtered_results)

            # Display Results in Table Format
            st.subheader(f"📖 Filtered Results for Keyword: '{keyword}'")
            st.dataframe(results_df)

            # Generate CSV data
            csv_buffer = BytesIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            # Generate PDF data
            pdf_buffer = create_pdf(st.session_state.filtered_results)

            # **Download Buttons in the Same Row**
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_buffer,
                    file_name="filtered_video_analysis.csv",
                    mime="text/csv"
                )

            with col2:
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_buffer,
                    file_name="filtered_video_summary.pdf",
                    mime="application/pdf"
                )

        # **Clear all stored session data only when explicitly requested**
        if clear_memory_clicked:
            st.session_state.filtered_results = []  # Clear only the results, not the whole session state
            st.rerun()  # Restart the app to reflect changes
