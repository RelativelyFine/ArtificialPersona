import sys
sys.path.append('../Wav2Lip')
import numpy as np
import cv2, os, sys, audio
import time
import torch, face_detection
from models import Wav2Lip
import tempfile
import subprocess, platform
from tqdm import tqdm
import queue
from pydub import AudioSegment
import io

class Wav2LipInference:
    def __init__(self, checkpoint_path, outfile='results/result_voice.mp4', resize_factor=1,
                 pads=[0, 10, 0, 0], face_det_batch_size=16, wav2lip_batch_size=128,
                 crop=[0, -1, 0, -1], box=[-1, -1, -1, -1], rotate=False, nosmooth=False, static=False, fps=30.0):
        self.checkpoint_path = checkpoint_path
        self.outfile = outfile
        self.resize_factor = resize_factor
        self.pads = pads
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.crop = crop
        self.box = box
        self.rotate = rotate
        self.nosmooth = nosmooth
        self.static = static
        self.fps = fps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = 96
        self.mel_step_size = 16
        self.model = self.load_model(self.checkpoint_path)
        print ("Model loaded")

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                flip_input=False, device=self.device)

        batch_size = self.face_det_batch_size
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break
        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                pass
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results 
    
    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(frames) # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
    
    def load_model(self, path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()
    
    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def inference(self, mp3_data, frame_queue):
        if mp3_data is not None:
            wav_bytes_file = mp3_data.getvalue()
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                    tmpfile.write(wav_bytes_file)
                    tmpfile.seek(0)
                    temp_file_path = tmpfile.name
                if temp_file_path:
                    wav = audio.load_wav(temp_file_path, 16000)
                    mel = audio.melspectrogram(wav)
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        else:
            raise ValueError("No audio data provided")

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        # Calculate mel chunks
        mel_chunks = []
        mel_idx_multiplier = 80. / self.fps  # Assuming self.fps is defined
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):  # Assuming self.mel_step_size is defined
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        # Process frames from the queue
        full_frames = []
        # while frame_queue.qsize() >= processing_queue_size:
        while not frame_queue.empty():
            frame = frame_queue.get()
            frame_queue.task_done()
            if self.resize_factor > 1:  # Assuming self.resize_factor is defined
                frame = cv2.resize(frame, (frame.shape[1] // self.resize_factor, frame.shape[0] // self.resize_factor))

            if self.rotate:  # Assuming self.rotate is defined
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = self.crop  # Assuming self.crop is defined
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

        # print(f"Number of frames available for inference: {len(full_frames)}")

        # Match the number of frames and mel chunks
        min_length = min(len(full_frames), len(mel_chunks))
        full_frames = full_frames[:min_length]
        mel_chunks = mel_chunks[:min_length]

        gen = self.datagen(full_frames.copy(), mel_chunks)  # Assuming datagen is correctly implemented
        processing_queue_temp = queue.Queue()
        try:
            for img_batch, mel_batch, frames, coords in gen:  # Assuming self.wav2lip_batch_size is defined
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

                with torch.no_grad():
                    pred = self.model(mel_batch, img_batch)

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                    f[y1:y2, x1:x2] = p
                    processing_queue_temp.put(f)
        except Exception as e:
            for frame in full_frames:
                processing_queue_temp.put(frame)

        return processing_queue_temp
