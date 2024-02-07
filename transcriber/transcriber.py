import speech_recognition as sr
import numpy as np
import noisereduce as nr
import nltk
import re
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from audio_utils import AudioUtils
from tts_service import TTSService

class Transcriber:
    def __init__(self, args):
        self.args = args
        self.audio_utils = AudioUtils(args)
        self.tts_service = TTSService(args)
        self.data_queue = Queue()
        self.transcription = ['']
        self.user_inputs_since_last_prompt = ['']
        self.buffered_messages = ''
        self.audio_queue = Queue()
        self.waitingAudios = 0
        self.previousQueueEmpty = True

    def start_transcribing(self):
        self.audio_utils.setup_recorder_and_microphone()
        self.audio_utils.recorder.listen_in_background(self.audio_utils.source, self.record_callback, phrase_time_limit=self.args.record_timeout)
        print("Model loaded.\nListening...")
        self.transcribe_loop()

    def transcribe_loop(self):
        recognizer = sr.Recognizer()
        while True:
            try:
                self.process_audio_data(recognizer)
                sleep(0.2)
            except KeyboardInterrupt:
                break
        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)

    def process_audio_data(self, recognizer):
        now = datetime.utcnow()
        phrase_timed_out = self.audio_utils.phrase_timed_out(now)

        if not self.data_queue.empty() or phrase_timed_out:
            self.previousQueueEmpty = False

            if phrase_timed_out:
                self.audio_utils.reset_last_sample()
                self.previousQueueEmpty = True

            self.audio_utils.update_phrase_time(now)
            audio_processed, audio_data = self.audio_utils.get_audio_data(self.data_queue)
            text = self.audio_utils.transcribe_audio(audio_processed, recognizer)
            self.process_transcription(text)

    def record_callback(self, _, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def process_transcription(self, text):
        if self.previousQueueEmpty:
            self.user_inputs_since_last_prompt.append(text)
            temp_prompt = self.user_inputs_since_last_prompt.copy()
            self.transcription.append(temp_prompt)
            self.user_inputs_since_last_prompt = ['']
            self.tts_service.deliver_to_tts(temp_prompt, self.audio_queue, self.waitingAudios)
        else:
            self.transcription[-1] = text
            self.user_inputs_since_last_prompt[-1] = text

