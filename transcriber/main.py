from arg_parser import parse_args
from transcriber import Transcriber
import wav2lip
import torch

def main():
    torch.cuda.init()
    args = parse_args()
    transcriber = Transcriber(args)
    transcriber.start_transcribing()
    wav2lip.wav2lip()

if __name__ == "__main__":
    main()
