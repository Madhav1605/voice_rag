import os
import collections
import queue
import time

import numpy as np
import sounddevice as sd
import webrtcvad

from scipy.io.wavfile import write
from faster_whisper import WhisperModel


model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

# 0 = least aggressive (lets noise through as "speech")
# 3 = most aggressive (best for noisy rooms / sensitive mics)
VAD_AGGRESSIVENESS = 3
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
FRAME_BYTES = FRAME_SIZE * 2 

MAX_WAIT_TIME = 10            
MIN_SPEECH_FRAMES = 5         
SILENCE_FRAMES_TO_STOP = 20   
PRE_ROLL_FRAMES = 5           


def speech_to_text():
    q = queue.Queue()
    def callback(indata, frames, callback_time, status):
        if status:
            print(f"Audio status: {status}")
        q.put(bytes(indata))
    print("\nListening...")
    start_time = time.time()
    speech_detected = False
    silence_frames = 0
    speech_frames = 0
    audio_chunks = []
    pre_roll = collections.deque(maxlen=PRE_ROLL_FRAMES)

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=callback
    ):
        max_frames = int((MAX_WAIT_TIME * 1000) / FRAME_DURATION_MS)
        for _ in range(max_frames):
            frame = q.get()
            if len(frame) != FRAME_BYTES:
                continue
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            if is_speech:
                speech_frames += 1
                silence_frames = 0
                if not speech_detected:
                    audio_chunks.extend(pre_roll)
                    speech_detected = True
                audio_chunks.append(np.frombuffer(frame, dtype=np.int16))
            elif speech_detected:
                silence_frames += 1
                audio_chunks.append(np.frombuffer(frame, dtype=np.int16))
                if silence_frames > SILENCE_FRAMES_TO_STOP:
                    break
            else:
                pre_roll.append(np.frombuffer(frame, dtype=np.int16))

    listening_time = time.time() - start_time
    print(f"Listening Time: {listening_time:.2f} sec")
    print(f"Speech Frames: {speech_frames}")
    if not speech_detected or speech_frames < MIN_SPEECH_FRAMES:
        print("No speech detected")
        return None
    audio = np.concatenate(audio_chunks)
    temp_path = "temp_stt.wav"
    write(temp_path, SAMPLE_RATE, audio)
    print("Transcribing...")
    segments, info = model.transcribe(
        temp_path,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False
    )
    text = " ".join(segment.text for segment in segments).strip()
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return text