import streamlit as st
import speech_recognition as sr
import os
import google.generativeai as genai
from gtts import gTTS
import pygame
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Initialize speech recognition and generative AI
recognizer = sr.Recognizer()
os.environ["GENAI_API_KEY"] = "AIzaSyDzIBbNBYEFFDaj_JLIXJrzJ37bjlQCa0k"
genai.configure(api_key=os.environ["GENAI_API_KEY"])
language = 'en'
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Define a custom video processor to capture webcam video
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Process the video frame (e.g., display in OpenCV or perform real-time analysis)
        return frame

# Streamlit app layout
st.title("Speech Recognition and Response")

# Webcam video capture
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

# Speech recognition and response
if webrtc_ctx.video_transformer:
    st.write("Webcam Video")
    st.image(webrtc_ctx.video_transformer.frame_out)

    with st.spinner("Waiting for audio input..."):
        with sr.Microphone() as source:
            audio_data = recognizer.listen(source)

        try:
            input_text = recognizer.recognize_google(audio_data)
            st.write("You said:", input_text)

            response = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [input_text],
                    },
                ]
            )
            response_text = response.text if hasattr(response, 'text') else str(response)
            st.write("Response:", response_text)

            myobj = gTTS(text=response_text, lang=language, slow=False)
            file_path = "welcome.mp3"
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except PermissionError:
                    st.write(f"PermissionError: Unable to delete {file_path}, file is in use.")
                myobj.save(file_path)
                pygame.mixer.init()
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                time.sleep(2)  # Wait for audio to finish playing
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except sr.UnknownValueError:
            st.write("Sorry, could not understand audio.")
        except sr.RequestError as e:
            st.write(f"Error: Could not request results from Google Speech Recognition service; {e}")
