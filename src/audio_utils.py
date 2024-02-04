import speech_recognition as sr
import torch
import numpy as np
import noisereduce as nr
from datetime import datetime, timedelta
from sys import platform
from transformers import AutoTokenizer, AutoModelForSequenceClassification



class AudioUtils:
    def __init__(self, args):
        self.args = args
        self.last_sample = bytes()
        self.phrase_time = None
        self.recorder = None
        self.source = None
        self.model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model')
        self.tokenizer = AutoTokenizer.from_pretrained('vectara/hallucination_evaluation_model')
        

    def setup_recorder_and_microphone(self):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.args.energy_threshold
        self.recorder.dynamic_energy_threshold = False
        self.source = self.setup_microphone()

    def setup_microphone(self):
        if 'linux' in platform:
            mic_name = self.args.default_microphone
            if not mic_name or mic_name == 'list':
                self.print_microphone_devices()
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        return sr.Microphone(sample_rate=48000, device_index=index)
        else:
            return sr.Microphone(sample_rate=48000)

    def phrase_timed_out(self, current_time):
        if self.phrase_time is None:
            return False
        return current_time - self.phrase_time > timedelta(seconds=self.args.phrase_timeout)

    def reset_last_sample(self):
        self.last_sample = bytes()

    def update_phrase_time(self, current_time):
        self.phrase_time = current_time

    def get_audio_data(self, data_queue):
        audio_data = []
        while not data_queue.empty():
            data, timestamp = data_queue.get()
            self.last_sample += data
            audio_data.append((data, timestamp))

        if self.last_sample:
            audio_numpy = np.frombuffer(self.last_sample, dtype=np.int16)
            reduced_noise = nr.reduce_noise(y=audio_numpy, sr=self.source.SAMPLE_RATE) if audio_numpy.size != 0 else audio_numpy
            audio_processed = sr.AudioData(reduced_noise.tobytes(), self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
            return audio_processed, audio_data
        return None, None

    def transcribe_audio(self, audio_data, recognizer):
        if audio_data is None:
            return ""
        try:
            transcription = recognizer.recognize_whisper(audio_data, model=self.args.model)

            # tokenize the transcription
            inputs = self.tokenizer.encode_plus(transcription, return_tensors='pt')

            # get the hallucination scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().detach().numpy()
                scores = 1 / (1 + np.exp(-logits)).flatten()


            # apply a thrshold to the scores to decide whether to keep or discard the current transciption
                
            if scores < 0.5:
                return ""
            else:
                return transcription
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Whisper service; {e}")
            return ""