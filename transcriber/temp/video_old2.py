import cv2
import time
from pydub import AudioSegment
import numpy as np
from pydub.generators import WhiteNoise
import io
from elevenlabs import play
from multiprocessing import Process, Queue, Event, Semaphore, Manager
from inference import Wav2LipInference

class Video:
    def __init__(self, checkpoint_path):
        manager = Manager()
        self.display_delay = 6  # Delay for displaying frames, in seconds
        self.processing_delay = 3  # Delay for processing frames, in seconds
        self.fps = 30
        self.display_queue_size = int(self.fps * self.display_delay)  # Queue size for display delay
        self.processing_queue_size = int(self.fps * self.processing_delay)  # Queue size for processing delay
        self.processing_list = manager.list()
        self.is_capturing = True
        self.queued_audio = Queue()
        self.queued_audio_playable = Queue()
        self.checkpoint_path = checkpoint_path
        self.inferencer = Wav2LipInference(checkpoint_path=self.checkpoint_path)

    def capture_frames(self, filled_event, consumed_event):
        frame_queue = Queue()
        cap = cv2.VideoCapture(0)  # Move the VideoCapture initialization here
        self.fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        processing_queue_curr_size = 0
        tmp = 0
        print("Hi! I'm in the capture frames")
        while self.is_capturing:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame_queue.put(frame)
            while processing_queue_curr_size < self.processing_queue_size and not frame_queue.empty():
                frame = frame_queue.get()
                self.processing_list.append(frame)
                processing_queue_curr_size += 1
            if processing_queue_curr_size >= self.processing_queue_size:
                filled_event.set()
                processing_queue_curr_size = 0
                consumed_event.wait()
                consumed_event.clear()

        cap.release()  # Make sure to release the capture here

    def display_frames(self, play_event, processed_frames_queue):
        print("Hi! I'm in the display frames")
        try:
            display_interval = 1 / self.fps
            next_frame_time = time.time() + display_interval
            frame_counter = self.fps

            while self.is_capturing:
                current_time = time.time()
                if current_time >= next_frame_time and not processed_frames_queue.empty():
                    try:
                        frame = processed_frames_queue.get(block=False)
                        cv2.imshow('Frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        if frame_counter == self.fps:
                            frame_counter = 0
                            play_event.set()
                        else:
                            frame_counter += 1
                    except Exception as e:
                        print(f"Error displaying frames: {e}")
                        time.sleep(0.01)

                    # Calculate the time for the next frame, adjusting for any processing delay
                    next_frame_time += display_interval
                    if next_frame_time < current_time:
                        # If we're running behind, skip frames to catch up
                        next_frame_time = current_time + display_interval

                # Adjust sleeping time to reduce CPU usage, but it's short enough not to affect timing accuracy.
                sleep_time = max(0, next_frame_time - time.time() - 0.005)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            print(f"Error displaying frames: {e}")


    def inference_loop(self, mp3_queue, processed_frames_queue, filled_event, consumed_event):
        print("Hi! I'm in the inference loop")
        while self.is_capturing:
            if not mp3_queue.empty():
                mp3 = mp3_queue.get()
            else:
                # make mp3 a display_delay long silence as a placeholder using pydub
                silent_mp3 = AudioSegment.silent(duration=self.processing_delay * 1000)
                # Export this silent segment to an MP3 format in memory
                mp3 = io.BytesIO()
                silent_mp3.export(mp3, format="mp3")
                mp3.seek(0)
                mp3 = mp3.getvalue()
            filled_event.wait()
            filled_event.clear()
            try:
                self.inferencer.inference(mp3, self.processing_list, processed_frames_queue)
                self.queued_audio.put(mp3)
            except Exception as e:
                print(f"Error processing frames: {e}")
            finally:
                self.processing_list[:] = []
                consumed_event.set()
            time.sleep(0.2)

    def stream_from_queue(self, play_event):
        print("Hi! I'm in the stream from queue")
        while True:
            play_event.wait()
            audio = self.queued_audio.get()
            if audio is not None:
                play(audio)
                self.queued_audio.task_done()
            time.sleep(0.1)

    def start_video(self, mp3_queue):
        play_event = Event()
        filled_event = Event()
        consumed_event = Event()
        processed_frames_queue = Queue()
        processes = [
            Process(target=self.capture_frames, args=(filled_event,consumed_event,)),
            Process(target=self.inference_loop, args=(mp3_queue,processed_frames_queue,filled_event,consumed_event,)),
            # Process(target=self.display_frames, args=(play_event,processed_frames_queue)),
            # Process(target=self.stream_from_queue, args=(play_event,))
        ]

        for p in processes:
            p.start()

        self.display_frames(play_event,processed_frames_queue)

        for p in processes:
            p.join()

        cv2.destroyAllWindows()
