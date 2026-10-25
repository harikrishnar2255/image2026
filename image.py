import os
import base64
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(page_title="Vision Chatbot", page_icon="🖼️", layout="centered")

st.title("🖼️ Vision Chatbot with Groq")
st.caption("Powered by Groq & `qwen/qwen3.8-27b`")

# Get the API key stored in variable 'x'
api_key = os.getenv("x")

if not api_key:
    st.error("API key 'x' not found in your .env file. Please check your setup.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=api_key)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Image uploader in the sidebar or main interface
uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

encoded_image_url = None
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Convert uploaded file to base64 data URL format required by Groq vision API
    bytes_data = uploaded_file.getvalue()
    base64_encoded = base64.b64encode(bytes_data).decode("utf-8")
    mime_type = uploaded_file.type
    encoded_image_url = f"data:{mime_type};base64,{base64_encoded}"

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle content blocks (text or image structures)
        content = message["content"]
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    st.markdown(item["text"])
                elif item.get("type") == "image_url":
                    st.image(item["image_url"]["url"])
        else:
            st.markdown(content)

# User prompt input
if prompt := st.chat_input("Ask something about the image or chat..."):
    # Build user message content payload
    user_content = [{"type": "text", "text": prompt}]
    
    # If an image is uploaded, attach it to the current message
    if encoded_image_url:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": encoded_image_url}
        })
    
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_content})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file:
            st.image(uploaded_file, width=200)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image..."):
            try:
                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=st.session_state.messages,
                    temperature=0.2
                )
                bot_reply = response.choices[0].message.content
                st.markdown(bot_reply)
                
                # Append assistant reply to session state
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"Error processing request: {e}")