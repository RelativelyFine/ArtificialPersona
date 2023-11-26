import argparse
import io
import os
import threading
from datetime import datetime, timedelta
import time
from queue import Queue
import speech_recognition as sr
import whisper
import torch

from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

class Transcriber:
    def __init__(self, args):
        # setting up all the arguments into class variables
        self.args = args
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        self.recorder.dynamic_energy_threshold = False
        self.data_queue = Queue()
        self.phrase_time = None
        self.last_sample = bytes()
        self.model = self.load_model(args.model, args.non_english)
        self.temp_file = NamedTemporaryFile().name
        self.output_file = open('hello.txt', 'w')
        self.transcription = ['']
        self.audio_buffer = []

    # load the model given the
    def load_model(self, model_name, non_english):
        if model_name != "large" and not non_english:
            model_name = model_name + ".en"
        return whisper.load_model(model_name)

    def get_microphone(self):
        if 'linux' in platform:
            mic_name = self.args.default_microphone
            if not mic_name or mic_name == 'list':
                print("Avaliable devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return None
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        return sr.Microphone(sample_rate=16000, device_index=index)
        else:
            return sr.Microphone(sample_rate=16000)

    def record_callback(self, recgonizer, audio: sr.AudioData):
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def generate_tts(self, buffer):
        """Function to send text to the TTS model."""
        if buffer:
            # Logic to send the buffer to the TTS model 'voicebox'
            print("Sending to TTS:", buffer)
            # voicebox.send(buffer) # Replace with actual TTS API call
        self.audio_buffer = []  # Reset the buffer

    def check_and_send_buffer(self):
        """Periodically check if the buffer needs to be sent to TTS."""
        while self.is_transcribing:
            if len(self.audio_buffer) >= 2 or (datetime.utcnow() - self.last_transcription_time).seconds >= 5:
                if self.audio_buffer:
                    threading.Thread(target=self.generate_tts, args=(self.audio_buffer[:],)).start()
                    self.audio_buffer = []
            time.sleep(5)  # Check every 5 seconds

    def transcribe_audio(self):
        self.is_transcribing = True
        self.last_transcription_time = datetime.utcnow()
        buffer_check_thread = threading.Thread(target=self.check_and_send_buffer)
        buffer_check_thread.start()
        while True:
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not self.data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.args.phrase_timeout):
                        self.last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    self.phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data

                    # Use AudioData to convert the raw data to wav data.
                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    # Write wav data to the temporary file as bytes.
                    with open(self.temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    # Read the transcription.
                    result = self.model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    self.update_and_send_to_buffer(text)

                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                    # Clear the console to reprint the updated transcription.
                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in self.transcription:
                        print(line)
                        self.source_file.write(line)

                    self.source_file.flush()
                    # Flush stdout.
                    print('', end='', flush=True)

                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except KeyboardInterrupt:
                self.source_file.close()
                break
        self.is_transcribing = False
        buffer_check_thread.join()    

        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)

    def start_transcription(self):
        self.source = self.get_microphone()
        if self.source is None:
            return

        with self.source as source:
            self.recorder.adjust_for_ambient_noise(source)
            self.recorder.listen_in_background(source, self.record_callback, phrase_time_limit=self.args.record_timeout)

        print("Model loaded\n")
        self.transcribe_audio()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    
