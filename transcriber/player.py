import os
import time
import threading
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from queue import Queue
import io
from video import Video
from inference import Wav2LipInference
import pyaudio

p = pyaudio.PyAudio()
stream = None

def play_wav(audio_data, p, stream):
    # Convert bytes data to a BytesIO object
    noisy_audio_segment = AudioSegment.from_file(audio_data, format="mp3")
    if stream is None:
        print(noisy_audio_segment.sample_width, noisy_audio_segment.channels,
        noisy_audio_segment.frame_rate)
        stream = p.open(format=p.get_format_from_width(noisy_audio_segment.sample_width),
            channels=noisy_audio_segment.channels,
            rate=noisy_audio_segment.frame_rate,
            output=True)
    raw = noisy_audio_segment.raw_data
    # if the length of the raw data is greater than processing_delay, trim it
    stream.write(raw)

# Function to add noise to silent audio and return as io.BytesIO
def add_noise_to_audio(file_path):
    silent_audio = AudioSegment.from_file(file_path, format="mp3")
    noise = WhiteNoise().to_audio_segment(duration=len(silent_audio))
    epsilon = 0.0001  # The level of noise to add
    noisy_audio = silent_audio.overlay(noise - 60, gain_during_overlay=epsilon)
    mp3_data_with_noise = io.BytesIO()
    noisy_audio.export(mp3_data_with_noise, format="mp3")
    return mp3_data_with_noise

# Function to continuously check the tmp directory and process new files
def process_audio_files(event, queue, p, stream, directory='tmp'):
    while not event.is_set():
        try:
            # List all mp3 files in the directory
            files = [f for f in os.listdir(directory) if f.endswith('.wav')]
            # Sort files from decreasing to increasing
            files.sort()
            for file_name in files:
                file_path = os.path.join(directory, file_name)
                # Process and add the file to the queue
                mp3_data_with_noise = add_noise_to_audio(file_path)
                # play_wav(mp3_data_with_noise, p, stream)
                queue.put(mp3_data_with_noise)
                # Optionally delete or move the file after processing
                os.remove(file_path)
        except Exception as e:
            print(f"Error processing files: {e}")
        time.sleep(0.2)  # Wait for 0.2 seconds before checking again

# Initialize the queue and event
mp3_data_queue = Queue()
stop_event = threading.Event()

# Start the thread to process audio files
processing_thread = threading.Thread(target=process_audio_files, args=(stop_event, mp3_data_queue, p, stream))
processing_thread.start()

checkpoint_path = 'wav2lip_gan.pth'
wav2lip_inferencer = Wav2LipInference(checkpoint_path=checkpoint_path)
video = Video(wav2lip_inferencer)
video.start_video(mp3_data_queue)

stop_event.set()
processing_thread.join()
stream.close()