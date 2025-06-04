import os
import librosa
import numpy as np
import soundfile as sf
import gc  # 导入垃圾回收器

def trim_silence(y, sr, top_db=15, frame_length=2048, hop_length=512):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, 
                                       frame_length=frame_length, 
                                       hop_length=hop_length)
    return y_trimmed

def random_slices(y, sr, num_slices=1, slice_duration=10):
    print("Creating random slices...")
    slice_samples = int(slice_duration * sr)
    if slice_samples > len(y):
        raise ValueError("音频长度不足以生成所需的切片。")
    slices = []
    for _ in range(num_slices): 
        start_sample = np.random.randint(0, len(y) - slice_samples)
        slice_y = y[start_sample:start_sample + slice_samples]
        slices.append(slice_y)
    return slices

def compute_mel_spectrogram(slices, output_path, sr=44100, n_mels=128):
    print("Computing mel spectrograms...")
    mel_spectrograms = []
    for slice_y in slices:
        mel_spectrogram = librosa.feature.melspectrogram(y=slice_y, sr=sr, n_mels=n_mels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrograms.append(mel_spectrogram)
    
    mel_spectrograms_array = np.array(mel_spectrograms, dtype='float32')
    mel_spectrograms_array = (mel_spectrograms_array - np.min(mel_spectrograms_array)) / (np.max(mel_spectrograms_array) - np.min(mel_spectrograms_array))
    np.save(output_path, mel_spectrograms_array)

    # 释放内存
    del mel_spectrograms, mel_spectrograms_array
    gc.collect()

if __name__ == "__main__":
    input_audio_path = "Separate/Sample/vocals.wav"  # 替换为包含音频文件夹的路径
    output_path = "data"  # 替换为保存梅尔频谱数组的路径
    y, sr = librosa.load(input_audio_path, sr=44100)
    y = trim_silence(y, sr,)
    slices = random_slices(y, sr)
    compute_mel_spectrogram(slices, output_path)