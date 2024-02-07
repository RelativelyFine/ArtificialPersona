from arg_parser import parse_args
from transcriber import Transcriber
import torch

def main():
    torch.cuda.init()
    args = parse_args()
    transcriber = Transcriber(args)
    transcriber.start_transcribing()

if __name__ == "__main__":
    main()
