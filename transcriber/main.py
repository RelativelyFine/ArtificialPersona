from arg_parser import parse_args
from transcriber import Transcriber

def main():
    args = parse_args()
    transcriber = Transcriber(args)
    transcriber.start_transcribing()

if __name__ == "__main__":
    main()
