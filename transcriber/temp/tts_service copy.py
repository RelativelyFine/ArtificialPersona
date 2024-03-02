from concurrent.futures import ThreadPoolExecutor
import nltk
import re
from elevenlabs import generate, play, save
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
from time import sleep
from datetime import datetime
from translator import translate
import soundfile as sf
import numpy as np
import pyaudio
from openai import OpenAI
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import io
import wave
import queue
import pyaudio
import tempfile
from video import Video
from inference import Wav2LipInference

class TTSService:
    def __init__(self, args):
        self.client = OpenAI(api_key=args.openai_api_key)
        self.args = args
        self.buffered_messages = ''
        self.waitingAudios = 0
        self.audio_queue = None
        self.chunked_mp3_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        # self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True)
        self.stream = None

    def deliver_to_tts(self, curr_transcription, audio_queue, waitingAudios):
        self.audio_queue = audio_queue
        self.waitingAudios = waitingAudios
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.generate_and_queue, curr_transcription)
            executor.submit(self.stream_from_queue)

    def generate_and_queue(self, curr_transcription):
        try:
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
        except Exception as e:
            print(f"Error generating audio: {e}")

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
                print(f"Error accumulating audio: {e}")
                break  # Exit loop if the queue is empty or any error occurs

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
            return accumulated_audio_bytes

        # If not exceeding, simply convert the whole segment back to bytes
        else:
            noise = WhiteNoise().to_audio_segment(duration=len(accumulated_segment))
            epsilon = 0.0001
            noisy_audio = accumulated_segment.overlay(noise - 60, gain_during_overlay=epsilon)
            accumulated_audio_bytes = io.BytesIO()
            noisy_audio.export(accumulated_audio_bytes, format="mp3")
            return accumulated_audio_bytes

    def stream_from_queue(self):
        # checkpoint_path = 'wav2lip_gan.pth'
        # wav2lip_inferencer = Wav2LipInference(checkpoint_path=checkpoint_path)
        # video = Video(wav2lip_inferencer)
        # video.start_video(self.chunked_mp3_queue)
        try:
            while True:
                # audio = self.accumulate_audio(self.audio_queue)
                audio = self.audio_queue.get()
                if audio is not None:
                    # save(audio, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav")
                    
                    self.chunked_mp3_queue.put(audio)

                    self.waitingAudios -= 1
                    if self.args.tts_model == "elevenlabs":
                        print("AKLISDjhakjsdhkshd")
                        self.play_wav(audio)
                    elif self.args.tts_model == "bark":
                        # Stream audio
                        audio_bytes = audio.tobytes()
                        self.stream.write(audio_bytes)

                        # save audio to disk
                        date= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        write_wav(f'bark_generation{date}.wav', SAMPLE_RATE, audio)
                    self.audio_queue.task_done()
                sleep(0.5)
        except Exception as e:
            print(f"Error streaming audio: {e}")


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

    def play_wav(self, audio_data):
        # Convert bytes data to a BytesIO object
        noisy_audio_segment = AudioSegment.from_file(audio_data, format="mp3")
        if self.stream is None:
            self.stream = self.p.open(format=self.p.get_format_from_width(noisy_audio_segment.sample_width),
                channels=noisy_audio_segment.channels,
                rate=noisy_audio_segment.frame_rate,
                output=True)
        # if len(noisy_audio_segment) > 2000:
        #     # Trim the audio to the first 2 seconds
        #     noisy_audio_segment = noisy_audio_segment[:2000]
        raw = noisy_audio_segment.raw_data
        # if the length of the raw data is greater than processing_delay, trim it
        self.stream.write(raw)