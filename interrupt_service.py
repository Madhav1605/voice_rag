import time
import keyboard

from voice_service import (
    is_tts_active,
    stop_text_to_speech
)

def watch_for_stop():
    while is_tts_active():

        if keyboard.is_pressed("s"):
            print("\nTTS Interrupted")

            stop_text_to_speech()
            return True

        time.sleep(0.05)
    return False