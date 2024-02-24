from tts_service import TTSService as ts
import sys
import cv2
import datetime
import speech_recognition as sr
import numpy as np

sys.path.insert(1, "./Wav2Lip/")
import inference

def wav2lip():
    video_feed = cv2.VideoCapture()
    
    frames = []
    start = datetime.time()
    
    while True:
        ret, frame = video_feed.read()
        frames.append((datetime.time(), frame))
        end = datetime.time()
        
        if ts.waiting_audios == -1:
            inference.wav2lip('Wav2Lip\checkpoints\wav2lip.pth', frames[1], ts.audio_bytes, outfile=f'results/result_voice.mp4', 
                    static=False, fps=25.0, pads=[0, 10, 0, 0], face_det_batch_size=16, 
                    wav2lip_batch_size=128, resize_factor=1, crop=[0, -1, 0, -1], box=[-1, -1, -1, -1], 
                    rotate=False, nosmooth=False)
            frames.clear()
        elif datetime.timedelta(start, end) >= 5000000:
            last_frames = []
            for frame in frames:
                if datetime.timedelta(frame[0], end) <= 1000000:
                    last_frames.append(frame)
                    
            empty_audio = sr.AudioData(np.zeros(16000, dtype=np.float32), 16000, 16000)

            inference.wav2lip('Wav2Lip\checkpoints\wav2lip.pth', last_frames[1], empty_audio, outfile='results/result_voice.mp4', 
                    static=False, fps=25.0, pads=[0, 10, 0, 0], face_det_batch_size=16, 
                    wav2lip_batch_size=128, resize_factor=1, crop=[0, -1, 0, -1], box=[-1, -1, -1, -1], 
                    rotate=False, nosmooth=False)
            frames.clear()
            last_frames.clear()
            start = end