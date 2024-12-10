
import numpy as np
import wave
from scipy.io.wavfile import write
import simpleaudio as sa

# Frequency and timing settings
DIT_DURATION = 0.1  # seconds
DAH_DURATION = 0.3  # seconds
FREQUENCY = 440  # Hz (tone frequency)
SAMPLE_RATE = 44100  # Samples per second

# Morse code dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
    ',': '--..--', '.': '.-.-.-', '?': '..--..', '/': '-..-.', 
    '-': '-....-', '(': '-.--.', ')': '-.--.-', ' ': '/'
}

# Generate a sine wave for Morse code tones
def generate_tone(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    return tone.astype(np.float32)

# Convert text to Morse code
def text_to_morse(text):
    morse_code = []
    for char in text.upper():
        if char in MORSE_CODE_DICT:
            morse_code.append(MORSE_CODE_DICT[char])
    return ' '.join(morse_code)

# Convert Morse code to audio and save as .wav
def morse_to_audio(morse_code, output_file):
    audio = np.array([], dtype=np.float32)
    for symbol in morse_code:
        if symbol == '.':
            audio = np.concatenate((audio, generate_tone(FREQUENCY, DIT_DURATION, SAMPLE_RATE)))
        elif symbol == '-':
            audio = np.concatenate((audio, generate_tone(FREQUENCY, DAH_DURATION, SAMPLE_RATE)))
        elif symbol == '/':
            audio = np.concatenate((audio, np.zeros(int(SAMPLE_RATE * 0.5))))  # Space between words
        audio = np.concatenate((audio, np.zeros(int(SAMPLE_RATE * 0.1))))  # Space between dots/dashes

    write(output_file, SAMPLE_RATE, audio)
    print(f"Morse code audio saved to {output_file}")

# Decode Morse code to text
def morse_decoder(morse_code):
    reverse_dict = {v: k for k, v in MORSE_CODE_DICT.items()}
    words = morse_code.split('/')
    decoded_message = []
    for word in words:
        letters = word.split()
        decoded_message.append(''.join(reverse_dict[letter] for letter in letters))
    return ' '.join(decoded_message)

# Main program
if __name__ == "__main__":
    # Input text
    input_text = input("Enter text to convert to Morse code: ")
    morse_code = text_to_morse(input_text)
    print(f"Morse Code: {morse_code}")

    # Convert Morse to audio
    output_wav = "morse_code.wav"
    morse_to_audio(morse_code, output_wav)

    # Playback
    print("Playing Morse code...")
    wave_obj = sa.WaveObject.from_wave_file(output_wav)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    # Decode Morse back to text
    decoded_text = morse_decoder(morse_code)
    print(f"Decoded Text: {decoded_text}")