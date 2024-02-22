from concurrent.futures import ThreadPoolExecutor
import nltk
import re
from elevenlabs import generate, stream
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
from time import sleep
from datetime import datetime
from translator import translate
import pyaudio
from openai import OpenAI

class TTSService:
    def __init__(self, args):
        self.client = OpenAI(api_key=args.openai_api_key)
        self.args = args
        self.buffered_messages = ''
        self.waitingAudios = 0
        self.audio_queue = None
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True)

    def deliver_to_tts(self, curr_transcription, audio_queue, waitingAudios):
        self.audio_queue = audio_queue
        self.waitingAudios = waitingAudios
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.generate_and_queue, curr_transcription)
            executor.submit(self.stream_from_queue)

    def generate_and_queue(self, curr_transcription):
        full_message = ''
        for chunk_message in curr_transcription:
            self.buffered_messages += chunk_message
            full_message += chunk_message
            sentences = nltk.sent_tokenize(self.buffered_messages)
            print(chunk_message, end='')

            if self.waitingAudios > 0:
                continue

            sentence = self.get_complete_sentence(sentences)
            if sentence is None:
                continue
            self.generate_translate_audio(sentence)

        if self.buffered_messages:
            self.generate_translate_audio(self.buffered_messages)
            self.buffered_messages = ''

        # Add a sentinel to signal we're done generating
        self.audio_queue.put(None)

    def get_complete_sentence(self, sentences):
        if len(sentences) >= 2 and self.is_sentence(sentences[-1]):
            sentence = sentences.pop(0)
            while sentences:
                sentence += sentences.pop(0)
            self.buffered_messages = ' '.join(sentences)
            return sentence
        return None

    def generate_translate_audio(self, text):
        if self.args.from_language != self.args.to_language:
            print(f"Translating from {self.args.from_language} to {self.args.to_language}")
            text = translate(self.args.from_language, self.args.to_language, text, self.client)
        if self.args.tts_model == "elevenlabs":
            audio = generate(
                api_key=self.args.eleven_labs_api_key,
                text=text,
                voice=self.args.elevenlabs_voice,
                model="eleven_multilingual_v2",
                stream=True
            )
        elif self.args.tts_model == "bark":
            audio = generate_audio(text, history_prompt=self.args.bark_language) #Bark audio 
        self.waitingAudios += 1
        self.audio_queue.put(audio)

    def stream_from_queue(self):
        while True:
            audio = self.audio_queue.get()
            if audio is None:  # We're done streaming
                break
            self.waitingAudios -= 1
            if self.args.tts_model == "elevenlabs":
                stream(audio)
            elif self.args.tts_model == "bark":
                # Stream audio
                audio_bytes = audio.tobytes()
                self.stream.write(audio_bytes)
                # stream(audio)

                # save audio to disk
                date= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                write_wav(f'bark_generation{date}.wav', SAMPLE_RATE, audio)

            self.audio_queue.task_done()
            sleep(0.2)

    def is_sentence(self, text):
        pattern = r'^[^.]*[^\.0-9][.!?]$'
        return re.match(pattern, text) is not None

    def cleanup_audio_resources(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.p is not None:
            self.p.terminate()
            self.p = None
