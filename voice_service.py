import os
import numpy as np
import sounddevice as sd
import threading
import re

tts_active = threading.Event()
from piper.voice import PiperVoice


def prepare_for_tts(text):
    """
    Remove Markdown formatting so TTS sounds natural.
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)       # *italic*
    text = re.sub(r"`(.*?)`", r"\1", text)         # `code`
    text = text.replace("#", "")
    text = text.replace("-", "")
    text = text.replace("•", "")
    return text.strip()
VOICE_PATH = os.path.join(
    "voices",
    "en_US-hfc_female-medium.onnx"
)

VOICE = PiperVoice.load(
    VOICE_PATH
)


def play_text_to_speech(text):
    text = prepare_for_tts(text)
    try:
        tts_active.set()

        print("\nSpeaking...")

        audio_chunks = []
        sample_rate = None

        for chunk in VOICE.synthesize(text):
            sample_rate = chunk.sample_rate
            audio_chunks.append(
                chunk.audio_float_array
            )

        audio = np.concatenate(audio_chunks)

        silence = np.zeros(
            int(0.5 * sample_rate),
            dtype=np.float32
        )

        audio = np.concatenate(
            [audio, silence]
        )

        sd.play(audio, sample_rate)
        sd.wait()

    except Exception as e:
        print(f"TTS Error: {e}")

    finally:
        tts_active.clear()

def stop_text_to_speech():
    try:
        sd.stop()
    finally:
        tts_active.clear()

def is_tts_active():
    return tts_active.is_set()