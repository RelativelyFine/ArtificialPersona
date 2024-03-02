import cv2
import queue
import threading
import time
from pydub import AudioSegment
import numpy as np
from pydub.generators import WhiteNoise
import io
from elevenlabs import play
from concurrent.futures import ThreadPoolExecutor

from inference import Wav2LipInference

class Video:
    def __init__(self, wav2lip_inferencer):
        self.cap = cv2.VideoCapture(0)
        self.display_delay = 4  # Delay for displaying frames, in seconds
        self.processing_delay = 2  # Delay for processing frames, in seconds
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        self.display_queue_size = int(self.fps * self.display_delay)  # Queue size for display delay
        self.processing_queue_size = int(self.fps * self.processing_delay)  # Queue size for processing delay
        self.frame_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.processed_frames_queue = queue.Queue()
        self.is_capturing = True
        self.queued_audio = queue.Queue()
        self.queued_audio_playable = queue.Queue()
        self.mp3 = None
        self.inferencer = wav2lip_inferencer

    def capture_frames(self):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            if self.frame_queue.qsize() > self.display_queue_size:
                self.frame_queue.get()  # Remove the oldest frame if full.
            self.frame_queue.put(frame)
            if self.processing_queue.qsize() < self.processing_queue_size:
                self.processing_queue.put(frame)

    def display_frames(self):
        # Initial wait removed since we now have a continuous flow of processed frames
        display_interval = 1 / self.fps
        last_time = time.time()
        frames_played = 0

        while self.is_capturing:
            if not self.processed_frames_queue.empty() and time.time() - last_time >= display_interval:
                last_time = time.time()
                frames_played += 1
                frame = self.processed_frames_queue.get()
                cv2.imshow('Frame', frame)
                if frames_played == self.fps:
                    self.queued_audio_playable.put(self.queued_audio.get())
                    frames_played = 0
            time.sleep(0.02)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_capturing = False
                break

    def inference_loop(self, mp3_queue):
        while self.is_capturing:
            if mp3_queue.qsize() > 0:
                mp3 = mp3_queue.get()
            else:
                # make mp3 a display_delay long silence as a placeholder using pydub
                silent_mp3 = AudioSegment.silent(duration=self.processing_delay * 1000)
                # Export this silent segment to an MP3 format in memory
                mp3 = io.BytesIO()
                silent_mp3.export(mp3, format="mp3")
                mp3.seek(0)
                mp3 = mp3.getvalue()
            self.queued_audio.put(mp3)
            if self.processing_queue.qsize() >= self.processing_queue_size:
                processed_frames = None
                try:
                    processed_frames = self.inferencer.inference(mp3, self.processing_queue)
                except Exception as e:
                    print(f"Error processing frames: {e}")
                if processed_frames is not None:
                    while not processed_frames.empty():
                        self.processed_frames_queue.put(processed_frames.get())
                else:
                    while not self.processing_queue.empty():
                        self.processed_frames_queue.put(self.processing_queue.get())
            time.sleep(0.2)

    def stream_from_queue(self):
        while True:
            audio = self.queued_audio.get()
            if audio is not None:
                play(audio)
                self.queued_audio.task_done()
            time.sleep(0.02)

    def start_video(self, mp3_queue):
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.submit(self.capture_frames)
            executor.submit(self.inference_loop, mp3_queue)
            executor.submit(self.display_frames)
            executor.submit(self.stream_from_queue)

        self.cap.release()
        cv2.destroyAllWindows()