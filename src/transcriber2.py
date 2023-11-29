import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform


class Transcriber:
    def __init__(self, model="tiny", non_english=False, energy_threshold=1000,
                 record_timeout=2, phrase_timeout=3, default_microphone=None):
        self.model = model if not non_english else model + ".en"
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.default_microphone = default_microphone
        self.audio_model = whisper.load_model(self.model)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy_threshold
        self.recorder.dynamic_energy_threshold = False
        self.data_queue = Queue()
        self.transcription = ['']
        self.phrase_time = None
        self.last_sample = bytes()

        self.setup_microphone()

    def setup_microphone(self):
        if 'linux' in platform and self.default_microphone:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if self.default_microphone in name:
                    self.source = sr.Microphone(sample_rate=16000, device_index=index)
                    return
        self.source = sr.Microphone(sample_rate=16000)

    def record_callback(self, _, audio: sr.AudioData):
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def start_recording(self):
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

    def transcribe(self):
        temp_file = NamedTemporaryFile().name
        print("Model loaded.\n")
        source_file = open('hello.txt', 'w')
        try:
            while True:
                now = datetime.utcnow()
                if not self.data_queue.empty():
                    phrase_complete = False
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                        self.last_sample = bytes()
                        phrase_complete = True
                    self.phrase_time = now

                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data

                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    with open(temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    result = self.audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in self.transcription:
                        print(line)
                        source_file.write(line)

                    source_file.flush()
                    sleep(0.25)
        except KeyboardInterrupt:
            source_file.close()
        finally:
            print("\n\nTranscription:")
            for line in self.transcription:
                print(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true', help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000, help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2, help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3, help="How much empty space between recordings before we consider it a new line in the transcription.", type=float)

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse', help="Default microphone name for SpeechRecognition. Run this with 'list' to view available Microphones.", type=str)
        args = parser.parse_args()
        default_microphone = args.default_microphone
    else:
        args = parser.parse_args()
        default_microphone = None

    transcriber = Transcriber(model=args.model, non_english=args.non_english,
                              energy_threshold=args.energy_threshold, record_timeout=args.record_timeout,
                              phrase_timeout=args.phrase_timeout, default_microphone=default_microphone)
    
    print("running")
    transcriber.start_recording()
    transcriber.transcribe()

if __name__ == "__main__":
    main()

