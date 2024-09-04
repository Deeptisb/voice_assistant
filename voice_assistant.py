import os
import subprocess
import sys
import speech_recognition as sr
from gtts import gTTS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime
import pygame

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

try:
    import speech_recognition as sr
except ImportError:
    install_package('SpeechRecognition')
    import speech_recognition as sr

try:
    from gtts import gTTS
except ImportError:
    install_package('gtts')
    from gtts import gTTS

try:
    import pygame
except ImportError:
    install_package('pygame')
    import pygame

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    install_package('torch')
    install_package('transformers')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=1)  # Adjust this index if needed

    with mic as source:
        print('Listening for commands...')
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print("You said: ", command)
        return command.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def text_to_speech(response_text):
    print(response_text)
    tts = gTTS(text=response_text, lang="en")
    tts.save("response.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()
    os.remove("response.mp3")

def get_dialogpt_response(input_text):
    # Encode the input text
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

    # Generate a response
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

def get_current_time():
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    return f"The current time is {current_time}."

def get_current_date():
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    return f"Today's date is {current_date}."

def main():
    text_to_speech("Hello, what can I do for you today?")
    while True:
        command = listen_for_command()
        if command:
            if "exit" in command:
                text_to_speech("Goodbye.")
                break
            elif "time" in command:
                response = get_current_time()
            elif "data" in command:
                response=get_current_date()
            else:
                response = get_dialogpt_response(command)
            text_to_speech(response)

if __name__ == '__main__':
    main()
