import speech_recognition as sr
from pydub import AudioSegment
import os
import torch
from transformers import pipeline

def transcribe_audio(file_name):
    try:
        audio = AudioSegment.from_file(file_name)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    try:
        # Export to WAV format
        wav_file_name = file_name.replace(file_name.split('.')[-1], 'wav')
        audio.export(wav_file_name, format="wav")
    except Exception as e:
        print(f"Error converting to WAV: {e}")
        return

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file_name) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data, language="lt-LT")
            text_file_name = file_name.replace(file_name.split('.')[-1], 'txt')
            with open(text_file_name, 'w') as text_file:
                text_file.write(transcription)
            print(f"Transcription saved to {text_file_name}")
            delete_wav_file(wav_file_name)
    except sr.UnknownValueError:
        print("Audio was not clear enough to transcribe.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


def delete_wav_file(wav_file_name):
    if os.path.exists(wav_file_name):
        try:
            os.remove(wav_file_name)
        except Exception:
            pass 

def summarize_file(file_name):
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist.")
        return

    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("summarization", model="LukasStankevicius/t5-base-lithuanian-news-summaries-175", device=device)

    try:
        with open(file_name, "r", encoding="utf-8") as file:
            input_text = file.read()

        summary = pipe(input_text, min_length=70, max_length=150, temperature=0.2, repetition_penalty=1.7, do_sample=True)

        with open("summary.txt", 'w') as text_file:
                text_file.write(summary[0]['summary_text'])
        print(f"Summary saved to summary.txt")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    # Set audio file name!
    file_name = "audio.m4a"
    if os.path.exists(file_name):
        print("working...")
        transcribe_audio(file_name)
        updated_name = file_name.replace(file_name.split('.')[-1], 'txt')
        summarize_file(updated_name)
    else:
        print(f"File {file_name} does not exist.")

input("Press Enter to exit...")










