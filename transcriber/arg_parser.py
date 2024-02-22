import argparse
import os
from dotenv import load_dotenv
from sys import platform

def parse_args():
    try:
        load_dotenv()
        eleven_labs = os.getenv("ELEVENLABS_API_KEY")
        if eleven_labs == "":
            eleven_labs = None
        openai = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        print(e)
        print("Error: Could not load .env file.")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--eleven_labs_api_key", default=eleven_labs, help="API key for Eleven Labs TTS", type=str)
    parser.add_argument("--openai_api_key", default=openai, help="API key for OpenAI", type=str)
    parser.add_argument("--model", default="medium", help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true', help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=300, help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=1.2, help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=2.4, help="How much empty space between recordings before we consider it a new line in the transcription.", type=float)
    parser.add_argument("--from_language", default="english", help="Language you wanna translate from.", type=str)
    parser.add_argument("--to_language", default="english", help="Language you wanna translate to.", type=str)
    parser.add_argument("--elevenlabs_voice", default="Rachel", help="Picking the voice of the responder", type=str)
    parser.add_argument("--tts_model", default="elevenlabs", help="Choices are \"elevenlabs\" and \"bark\"", type=str)
    parser.add_argument("--bark_language", default="v2/en_speaker_6", help="Go to https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c to see presets.", type=str)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse', help="Default microphone name for SpeechRecognition. Run this with 'list' to view available Microphones.", type=str)
    return parser.parse_args()
