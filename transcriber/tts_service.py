from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pipe
import nltk
import re
from elevenlabs import generate, stream, save
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
from time import sleep
from datetime import datetime
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from translator import translate
import pyaudio
import io
from openai import OpenAI
from video import Video
from inference import Wav2LipInference
import queue

class TTSService:
    def __init__(self, args):
        self.client = OpenAI(api_key=args.openai_api_key)
        self.args = args
        self.buffered_messages = ''
        self.waitingAudios = 0
        self.audio_queue = queue.Queue()
        # self.p = pyaudio.PyAudio()
        # self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True)
        # self.stream = None


    def deliver_to_tts(self, curr_transcription, audio_queue, waitingAudios):
        self.audio_queue = audio_queue
        self.waitingAudios = waitingAudios
        with ThreadPoolExecutor(max_workers=2) as thread_executor:
            thread_executor.submit(self.stream_from_queue)
            thread_executor.submit(self.generate_and_queue, curr_transcription)

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
            print(f"  (Translating from {self.args.from_language} to {self.args.to_language})\n")
            text = translate(self.args.from_language, self.args.to_language, text, self.client)
        if self.args.tts_model == "elevenlabs":
            audio = generate(
                api_key=self.args.eleven_labs_api_key,
                text=text,
                voice=self.args.elevenlabs_voice,
                model="eleven_multilingual_v2",
                stream=False
            )
        elif self.args.tts_model == "bark":
            audio = generate_audio(text, history_prompt=self.args.bark_language) #Bark audio 
        self.waitingAudios += 1
        self.audio_queue.put(audio)

    def accumulate_audio(self, audio_queue, desired_duration=2.0):
        """
        Accumulate audio from the queue until reaching the desired duration (in seconds),
        properly handling WAV file format. If the accumulated audio exceeds the desired duration,
        trim the excess and optionally put it back into the queue.
        """

        accumulated_segment = AudioSegment.empty()
        total_duration_ms = 0
        while total_duration_ms < desired_duration * 1000:  # Convert seconds to milliseconds
            try:
                audio_bytes = audio_queue.get_nowait()
                self.waitingAudios -= 1
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                accumulated_segment += audio_segment
                total_duration_ms = len(accumulated_segment)
            except Exception as e:
                if not total_duration_ms > 0:
                    return None
                break  # Exit loop if the queue is empty or any error occurs
            finally:
                audio_queue.task_done()

        # If accumulated audio exceeds the desired duration, trim and return the excess to the queue
        if total_duration_ms > desired_duration * 1000:
            # Split the audio at the desired duration point
            split_point_ms = desired_duration * 1000
            trimmed_audio = accumulated_segment[:split_point_ms]
            excess_audio = accumulated_segment[split_point_ms:]

            # Optionally, encode excess audio back to bytes and put it back into the queue
            excess_audio_bytes = io.BytesIO()
            excess_audio.export(excess_audio_bytes, format="mp3")
            audio_queue.put(excess_audio_bytes.getvalue())
            self.waitingAudios += 1

            # Convert trimmed audio back to bytes
            noise = WhiteNoise().to_audio_segment(duration=len(trimmed_audio))
            epsilon = 0.0001
            noisy_audio = trimmed_audio.overlay(noise - 60, gain_during_overlay=epsilon)
            accumulated_audio_bytes = io.BytesIO()
            noisy_audio.export(accumulated_audio_bytes, format="mp3")
            return accumulated_audio_bytes.getvalue()

        # If not exceeding, simply convert the whole segment back to bytes
        else:
            noise = WhiteNoise().to_audio_segment(duration=len(accumulated_segment))
            epsilon = 0.0001
            noisy_audio = accumulated_segment.overlay(noise - 60, gain_during_overlay=epsilon)
            accumulated_audio_bytes = io.BytesIO()
            noisy_audio.export(accumulated_audio_bytes, format="mp3")
            return accumulated_audio_bytes.getvalue()

    def stream_from_queue(self):
        while True:
            audio = self.accumulate_audio(self.audio_queue)
            if audio is not None:
                if self.args.tts_model == "elevenlabs":
                    # self.play_wav(audio)
                    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                    with open(f"tmp/{curr_time}.wav", "wb") as f:
                        f.write(audio)
                elif self.args.tts_model == "bark":
                    # Stream audio
                    audio_bytes = audio.tobytes()
                    self.stream.write(audio_bytes)
                    # stream(audio)

                    # save audio to disk
                    date= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    write_wav(f'bark_generation{date}.wav', SAMPLE_RATE, audio)
            self.audio_queue.task_done()
            sleep(0.01)

    def is_sentence(self, text):
        pattern = r'^[^.]*[^\.0-9][.!?]$'
        return re.match(pattern, text) is not None

    def cleanup_audio_resources(self):
        pass