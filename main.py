import streamlit as st
from vlm_video.vlm_video import vlm_gbt
from utils.utils import load_restricted_words


def main():
    # Add tilted text above the navigation
    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 64px;
            font-weight: bold;
            color: orange;
            transform: rotate(0deg);
            margin-bottom: 10px;">
            Vision-Language Model
        </div>
        """,
        unsafe_allow_html=True,
    )

    # st.sidebar.title("Navigation")
    st.sidebar.markdown(
        """
        <div style="
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: Black;
            margin-bottom: 10px;">
            Navigation
        </div>
        """,
        unsafe_allow_html=True,
        )

    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

    restricted_list, model_list = load_restricted_words()
    base_model = st.sidebar.selectbox("Select Base Model for Fine-Tuning", options=model_list)

    choice = st.sidebar.radio("Go to", ["VLM_video"])

    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API Key to proceed.")
        return

    if choice == "VLM_video" and base_model == "gpt-4-turbo":
        vlm_gbt(api_key, base_model)


if __name__ == "__main__":
    main()
