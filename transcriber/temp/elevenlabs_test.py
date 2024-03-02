str = "dba606f2393682635870328d545a9ff7"

from elevenlabs import generate, play

audio = generate(
  api_key="dba606f2393682635870328d545a9ff7",
  text="Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!",
  voice="Rachel",
  model="eleven_multilingual_v2"
)

play(audio)