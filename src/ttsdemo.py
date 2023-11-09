
from elevenlabs import generate, play
# import socks
# import socket
# import elevenlabs


# socks.set_default_proxy(socks.SOCKS5, "localhost", 9150)
# socket.socket = socks.socksocket

audio = generate (
  # api_key=eleven_labs,
  text="testing123",
  voice="Bella",
  model="eleven_monolingual_v1",
)
play(audio)