from arg_parser import parse_args
from multiprocessing import Pipe
from transcriber import Transcriber
import torch
import subprocess
from concurrent.futures import ProcessPoolExecutor
from video import Video
from inference import Wav2LipInference

def main():
    torch.cuda.init()
    args = parse_args()
    transcriber = Transcriber(args)
    transcriber.start_transcribing()

if __name__ == "__main__":
    main()
