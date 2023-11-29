import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import threading
import time

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

class Transcriber:
    def __init__(self, args):
        self.args = args
        self.model = self.load_model(args.model, args.non_english)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        self.recorder.dynamic_energy_threshold = False
        self.data_queue = Queue()
        self.phrase_time = None
        self.last_sample = bytes()
        self.temp_file = NamedTemporaryFile().name
        self.transcription = ['']
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.is_transcribing = False
        self.source = self.get_microphone()
        self.phrase_time = datetime.utcnow()

    def load_model(self, model_name, non_english):
        if model_name != "large" and not non_english:
            model_name = model_name + ".en"
        return whisper.load_model(model_name)

    def get_microphone(self):
        if 'linux' in platform and self.args.default_microphone:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if self.args.default_microphone in name:
                    return sr.Microphone(sample_rate=16000, device_index=index)
        return sr.Microphone(sample_rate=16000)

    def record_callback(self, recognizer, audio: sr.AudioData):
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def transcribe(self):
        self.is_transcribing = True
        while self.is_transcribing:
            now = datetime.utcnow()
            if not self.data_queue.empty():
                phrase_complete = False
                if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.args.phrase_timeout):
                    self.last_sample = bytes()
                    phrase_complete = True
                self.phrase_time = now

                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    self.last_sample += data

                audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                with open(self.temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                result = self.model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                with self.buffer_lock:
                    if phrase_complete:
                        self.audio_buffer.append(text)
                    else:
                        if self.audio_buffer:
                            self.audio_buffer[-1] = text  # Update the last phrase
                        else:
                            self.audio_buffer.append(text)  # Add the first phrase

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in self.transcription:
                    print(line)

                sleep(0.25)
    
    def manage_buffer(self):
        while self.is_transcribing or self.audio_buffer:
            with self.buffer_lock:
                if len(self.audio_buffer) >= 2 or (datetime.utcnow() - self.phrase_time).seconds >= 5:
                    self.send_to_tts()
            time.sleep(1)

    def send_to_tts(self):
        # Process the new phrases and then clear the buffer
        print("Sending to TTS:", self.audio_buffer)
        self.audio_buffer.clear()

    def start_transcription(self):
        with self.source as source:
            self.recorder.adjust_for_ambient_noise(source, duration=1)
        
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.args.record_timeout)

        print("Model loaded.\nStarting transcription...")
        self.is_transcribing = True
        transcription_thread = threading.Thread(target=self.transcribe)
        buffer_thread = threading.Thread(target=self.manage_buffer)

        transcription_thread.start()
        buffer_thread.start()

        transcription_thread.join()
        buffer_thread.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true', help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000, help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2, help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3, help="How much empty space between recordings before we consider it a new line in the transcription.", type=float)

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse', help="Default microphone name for SpeechRecognition. Run this with 'list' to view available Microphones.", type=str)

    args = parser.parse_args()

    transcriber = Transcriber(args)
    print("Running...")
    transcriber.start_transcription()

if __name__ == "__main__":
    main()
