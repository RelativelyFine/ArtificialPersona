import argparse
import platform
from speech_to_text_class import Transcriber  

def main():
    # Define arguments for the Transcriber class
    args = argparse.Namespace(
        model="tiny",
        non_english=False,
        energy_threshold=1000,
        record_timeout=2,
        phrase_timeout=3,
        default_microphone='pulse' if 'linux' in platform.system().lower() else None
    )

    # Create an instance of Transcriber
    print("Working")
    transcriber = Transcriber(args)

    # Start transcription
    transcriber.start_transcription()

if __name__ == "__main__":
    main()
