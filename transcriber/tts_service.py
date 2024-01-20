from concurrent.futures import ThreadPoolExecutor
import nltk
import re
from elevenlabs import generate, stream

class TTSService:
    def __init__(self, args):
        self.args = args
        self.buffered_messages = ''
        self.waitingAudios = 0
        self.audio_queue = None

    def deliver_to_tts(self, curr_transcription, audio_queue, waitingAudios):
        self.audio_queue = audio_queue
        self.waitingAudios = waitingAudios
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.generate_and_queue, curr_transcription)
            executor.submit(self.stream_from_queue)

    def generate_and_queue(self, curr_transcription):
        full_message = ''
        for chunk_message in curr_transcription:
            self.buffered_messages += chunk_message
            full_message += chunk_message
            sentences = nltk.sent_tokenize(self.buffered_messages)
            print(chunk_message, end='')

            if self.waitingAudios > 0:
                continue

            sentence = self.get_complete_sentence(sentences)
            if sentence:
                self.generate_audio(sentence)

        if self.buffered_messages:
            self.generate_audio(self.buffered_messages)
            self.buffered_messages = ''

        # Add a sentinel to signal we're done generating
        self.audio_queue.put(None)

    def get_complete_sentence(self, sentences):
        if len(sentences) >= 2 and self.is_sentence(sentences[-1]):
            sentence = sentences.pop(0)
            while sentences:
                sentence += sentences.pop(0)
            self.buffered_messages = ' '.join(sentences)
            return sentence
        return None

    def generate_audio(self, text):
        audio = generate(
            api_key=self.args.eleven_labs_api_key,
            text=text,
            voice="Glinda",
            model="eleven_monolingual_v1",
            stream=True
        )
        self.waitingAudios += 1
        self.audio_queue.put(audio)

    def stream_from_queue(self):
        while True:
            audio = self.audio_queue.get()
            if audio is None:  # We're done streaming
                break
            self.waitingAudios -= 1
            stream(audio)
            self.audio_queue.task_done()

    def is_sentence(self, text):
        pattern = r'^[^.]*[^\.0-9][.!?]$'
        return re.match(pattern, text) is not None