import os
import librosa
import numpy as np
import soundfile as sf

def trim_silence(y, sr, top_db=8, frame_length=2048, hop_length=512):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    sf.write("trimmed.wav", y_trimmed, sr)
    print("去除静音部分的音频已保存为 trimmed.wav")
    return y_trimmed

if __name__ == "__main__":
    y, sr = librosa.load("C:\Data\.Vocal\KAFU\KAFU (2).wav", sr=44100)
    trim_silence(y, sr)