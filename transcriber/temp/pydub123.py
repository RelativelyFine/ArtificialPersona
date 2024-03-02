from pydub import AudioSegment
from elevenlabs import generate, play, save
import io
from video import Video
from inference import Wav2LipInference
from pydub.generators import WhiteNoise
from multiprocessing import Process, Queue, Event
import wave
import pyaudio


# load in the audio file
# song = pydub.AudioSegment.from_file("barl.wav", format="wav")
# audio_segment = AudioSegment.from_file("accumulated_audio2024-03-01_19-21-45.wav", format="mp3")

# # Create a bytes buffer
# buffer = io.BytesIO()

# # Export the AudioSegment back to bytes (MP3 format in this example)
# audio_segment.export(buffer, format="mp3")

# # Get the bytes
# mp3_data = buffer.getvalue()

# Assuming you have your silent MP3 data in mp3_bytes
# Let's convert it to a pydub AudioSegment first
silent_audio = AudioSegment.from_file("output.mp3", format="mp3")

# Generate a short segment of white noise
noise = WhiteNoise().to_audio_segment(duration=len(silent_audio))

# The level of noise to add (epsilon). This value might need tweaking.
# It should be very low to not be audible but still present.
epsilon = 0.0001

# Mix the silent audio with the noise at a very low level
noisy_audio = silent_audio.overlay(noise - 60, gain_during_overlay=epsilon)

# If you want to export this noisy audio to MP3 format
mp3_data_with_noise = io.BytesIO()
noisy_audio.export(mp3_data_with_noise, format="mp3")

# Create a queue and put the MP3 data into it
mp3_data_queue = Queue()
mp3_data_queue.put(mp3_data_with_noise)
mp3_data_queue.put(mp3_data_with_noise)
mp3_data_queue.put(mp3_data_with_noise)
mp3_data_queue.put(mp3_data_with_noise)
mp3_data_queue.put(mp3_data_with_noise)
mp3_data_queue.put(mp3_data_with_noise)
mp3_data_queue.put(mp3_data_with_noise)
mp3_data_queue.put(mp3_data_with_noise)

checkpoint_path = 'wav2lip_gan.pth'
wav2lip_inferencer = Wav2LipInference(checkpoint_path=checkpoint_path)
video = Video(wav2lip_inferencer)
video.start_video(mp3_data_queue)