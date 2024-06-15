import streamlit as st
import cv2
import numpy as np
import speech_recognition as sr
import os
import google.generativeai as genai
from gtts import gTTS
import pygame
import time
import threading

os.environ["GENAI_API_KEY"] = "AIzaSyDzIBbNBYEFFDaj_JLIXJrzJ37bjlQCa0k"
genai.configure(api_key=os.environ["GENAI_API_KEY"])
language = 'en'
vid = cv2.VideoCapture(0)

recognizer = sr.Recognizer()

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "You are a Mock Interviewer. Ask the candidate questions and reply accordingly based on his/her career choice. Begin with asking him/her the Career option.",
            ],
        },
    ]
)

input_text = ""
response_text = ""
input_text_timer = None
response_text_timer = None
lock = threading.Lock()

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
    pygame.mixer.quit()

def speech_recognition():
    global input_text, response_text, input_text_timer, response_text_timer
    while True:
        with sr.Microphone() as source:
            st.write("Speak something...")
            audio_data = recognizer.listen(source)

        try:
            input_text = recognizer.recognize_google(audio_data)
            st.write("You said:", input_text)
        except sr.UnknownValueError:
            st.write("Sorry, could not understand audio.")
            continue
        except sr.RequestError as e:
            st.write("Error: Could not request results from Google Speech Recognition service; {0}".format(e))
            continue

        response = chat_session.send_message(input_text)
        response_text = response.text if hasattr(response, 'text') else str(response)
        st.write("Response:", response_text)

        myobj = gTTS(text=response_text, lang=language, slow=False)
        file_path = "welcome.mp3"
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                st.write(f"PermissionError: Unable to delete {file_path}, file is in use.")
                continue
        myobj.save(file_path)
        play_audio(file_path)
        input_text_timer = threading.Timer(3, reset_input_text)
        input_text_timer.start()
        response_text_timer = threading.Timer(3, reset_response_text)
        response_text_timer.start()

def reset_input_text():
    global input_text
    input_text = ""

def reset_response_text():
    global response_text
    response_text = ""

def draw_text(frame, text, pos, font_scale=1, font_thickness=2, max_words_per_line=6, color=(255, 255, 255)):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line) + len(word) <= max_words_per_line:
            line += word + " "
        else:
            lines.append(line)
            line = word + " "
    lines.append(line)
    y = pos[1]
    for line in lines:
        cv2.putText(frame, line.strip(), (pos[0], y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
        y += int(1.5 * font_scale * 20)

def main():
    speech_thread = threading.Thread(target=speech_recognition)
    speech_thread.daemon = True
    speech_thread.start()

    while True:
        ret, frame = vid.read()
        with lock:
            if input_text:
                draw_text(frame, f"You said: {input_text}", (10, 50), font_scale=0.5, color=(255, 0, 0))
            if response_text:
                draw_text(frame, f"Response: {response_text}", (10, 100), font_scale=0.5, color=(0, 255, 0))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
