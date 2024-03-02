import speech_recognition as sr
import numpy as np
import os
from datetime import datetime
from queue import Queue
from time import sleep
from audio_utils import AudioUtils
from tts_service import TTSService
from bark import preload_models

class Transcriber:
    def __init__(self, args):
        self.args = args
        self.tts_service = TTSService(args)
        self.data_queue = Queue()
        self.transcription = ['']
        self.user_inputs_since_last_prompt = ['']
        self.buffered_messages = ''
        self.audio_queue = Queue()
        self.waitingAudios = 0
        self.previousQueueEmpty = True
        self.audio_utils = AudioUtils(self.args)
        self.recognizer = sr.Recognizer()
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"

    def start_transcribing(self):
        preload_models()
        audio_bytes = sr.AudioData(np.zeros(16000, dtype=np.float32), 16000, 1)
        self.recognizer.recognize_whisper(audio_bytes, model=self.args.model)
        print("Model loaded.")
        self.audio_utils.setup_recorder_and_microphone()
        self.audio_utils.recorder.listen_in_background(self.audio_utils.source, self.record_callback, phrase_time_limit=self.args.record_timeout)
        print("Listening...")
        self.transcribe_loop()

    def transcribe_loop(self):
        while True:
            try:
                self.process_audio_data()
                sleep(0.2)
            except KeyboardInterrupt:
                break
        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)
        self.tts_service.cleanup_audio_resources()

    def process_audio_data(self):
        now = datetime.utcnow()
        phrase_timed_out = self.audio_utils.phrase_timed_out(now)
        if not self.data_queue.empty() or phrase_timed_out:
            self.previousQueueEmpty = False
            if phrase_timed_out:
                self.audio_utils.reset_last_sample()
                self.previousQueueEmpty = True

            self.audio_utils.update_phrase_time(now)
            audio_data, audio_processed = self.audio_utils.get_audio_data(self.data_queue)
            text = self.audio_utils.transcribe_audio(audio_processed, self.recognizer)
            self.process_transcription(text)

    def record_callback(self, _, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def process_transcription(self, text):
        if self.previousQueueEmpty:
            self.user_inputs_since_last_prompt.append(text)
            curr_transcription = self.user_inputs_since_last_prompt.copy()
            self.transcription.append(curr_transcription)
            self.user_inputs_since_last_prompt = ['']
            # os.system('cls' if os.name=='nt' else 'clear')
            # for line in self.transcription:
            #     print(line)
            # # Flush stdout.
            # print('', end='', flush=True)
            self.tts_service.deliver_to_tts(curr_transcription, self.audio_queue, self.waitingAudios)
        else:
            self.transcription[-1] = text
            self.user_inputs_since_last_prompt[-1] = text

