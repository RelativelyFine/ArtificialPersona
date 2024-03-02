import cv2
import queue
import threading
import time
from pydub import AudioSegment
import numpy as np
from pydub.generators import WhiteNoise
import io
import subprocess
from elevenlabs import play
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from inference import Wav2LipInference
import pyaudio
import wave

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
        self.inferencer = wav2lip_inferencer
        self.processing_queue_lock = threading.Lock()

    def capture_frames(self, consume_event):
        print("Hi! I'm in the capture frames")
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?).")
                continue
            # self.frame_queue.put(frame)
            # while self.processing_queue.qsize() < self.processing_queue_size and not self.frame_queue.empty():
            #     self.processing_queue.put(self.frame_queue.get())
            self.processing_queue.put(frame)
            if self.processing_queue.qsize() >= self.processing_queue_size:
                consume_event.set()
            # time.sleep(1/self.fps)

    def inference_loop(self, mp3_queue, consume_event, p, stream):
        try:
            print("Hi! I'm in the inference loop")
            while self.is_capturing:
                time.sleep(0.01)
                consume_event.wait()
                consume_event.clear()
                if not mp3_queue.empty():
                    mp3 = mp3_queue.get()
                else:
                    # make mp3 a display_delay long silence as a placeholder using pydub
                    silent_mp3 = AudioSegment.silent(duration=self.processing_delay * 1000)
                    # Export this silent segment to an MP3 format in memory
                    mp3 = io.BytesIO()
                    silent_mp3.export(mp3, format="mp3")
                mp3.seek(0)
                try:
                    newFrames = self.inferencer.inference(mp3, self.processing_queue)
                    mp3.seek(0)
                    if not newFrames.empty():
                        frame = newFrames.get()
                        self.processed_frames_queue.put([frame, mp3])
                    while not newFrames.empty():
                        frame = newFrames.get()
                        self.processed_frames_queue.put([frame, None])
                except Exception as e:
                    print(f"Error processing frames: {e}")
        except Exception as e:
            print(f"Error in inference loop: {e}")

    # def display_frames(self, play_event):
    #     print("Hi! I'm in the display frames")
    #     try:
    #         display_interval = 1 / self.fps
    #         delay_queue = queue.Queue()
    #         delay_time = 0.28
    #         always_take = False
    #         delay_frames = 0
    #         next_frame_time = time.time() + display_interval
    #         frame_counter = self.fps * self.processing_delay

    #         while self.is_capturing:
    #             current_time = time.time()
    #             if not self.processed_frames_queue.empty() and current_time >= next_frame_time:
    #                 incoming_frame_and_audio = self.processed_frames_queue.get()
    #                 self.processed_frames_queue.task_done()
    #                 audio = incoming_frame_and_audio[1]
    #                 frame = incoming_frame_and_audio[0]
    #                 delay_queue.put(frame)
    #                 if audio is not None:
    #                     print("Playing audio")
    #                     self.queued_audio.put(audio)
    #                 if always_take:
    #                     cv2.imshow('Frame', delay_queue.get())
    #                     cv2.waitKey(1) 
    #                 else:
    #                     always_take = delay_queue.qsize() > delay_frames

    #                 # Calculate the time for the next frame, adjusting for any processing delay
    #                 next_frame_time += display_interval
    #                 if next_frame_time < current_time:
    #                     # If we're running behind, skip frames to catch up
    #                     next_frame_time = current_time + display_interval

    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 self.is_capturing = False
    #                 break

    #             # Adjust sleeping time to reduce CPU usage, but it's short enough not to affect timing accuracy.
    #             sleep_time = max(0, next_frame_time - time.time() - 0.005)
    #             if sleep_time > 0:
    #                 time.sleep(sleep_time)
    #     except Exception as e:
    #         print(f"Error displaying frames: {e}")
    def display_frames(self, play_event):
        print("Hi! I'm in the display frames")
        try:
            display_interval = 1 / self.fps
            next_frame_time = time.time() + display_interval

            while self.is_capturing:
                current_time = time.time()
                if not self.processed_frames_queue.empty() and current_time >= next_frame_time:
                    incoming_frame_and_audio = self.processed_frames_queue.get()
                    self.processed_frames_queue.task_done()
                    audio = incoming_frame_and_audio[1]
                    frame = incoming_frame_and_audio[0]
                    cv2.imshow('Frame', frame)
                    cv2.waitKey(1) 
                    if audio is not None:
                        self.queued_audio.put(audio)

                    # Calculate the time for the next frame, adjusting for any processing delay
                    next_frame_time += display_interval
                    if next_frame_time < current_time:
                        # If we're running behind, skip frames to catch up
                        next_frame_time = current_time + display_interval

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_capturing = False
                    break

                # Adjust sleeping time to reduce CPU usage, but it's short enough not to affect timing accuracy.
                sleep_time = max(0, next_frame_time - time.time() - 0.005)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            print(f"Error displaying frames: {e}")

    def stream_from_queue(self, play_event, p, stream):
        print("Hi! I'm in the stream from queue")
        while True:
            audio = self.queued_audio.get()
            self.queued_audio.task_done()
            if audio is not None:
                self.play_wav(audio, p, stream)
            time.sleep(0.01)

    def start_video(self, mp3_queue):
        play_event = Event()
        consume_event = Event()
        p = pyaudio.PyAudio()
        stream = None
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.submit(self.capture_frames, consume_event)
            executor.submit(self.inference_loop, mp3_queue, consume_event, p, stream)
            executor.submit(self.display_frames, play_event)
            executor.submit(self.stream_from_queue, play_event, p, stream)

        self.cap.release()
        cv2.destroyAllWindows()

    def play_wav(self, audio_data, p, stream):
        # Convert bytes data to a BytesIO object
        noisy_audio_segment = AudioSegment.from_file(audio_data, format="mp3")
        if stream is None:
            stream = p.open(format=p.get_format_from_width(2),
                channels=1,
                rate=44100,
                output=True)
        raw = noisy_audio_segment.raw_data
        # if the length of the raw data is greater than processing_delay, trim it
        stream.write(raw)