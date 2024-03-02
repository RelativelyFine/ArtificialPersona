from pydub import AudioSegment

def trim_audio(input_file, output_file, end_time_ms=10000):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_file)
    
    # Trim the first 10 seconds (10000 milliseconds)
    trimmed_audio = audio[:end_time_ms]
    
    # Export the trimmed audio to a new file
    trimmed_audio.export(output_file, format="mp3")

# Example usage
input_file = './input.mp3'
output_file = './output.mp3'
trim_audio(input_file, output_file)