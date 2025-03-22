import os
import librosa
import numpy as np
from spleeter.separator import Separator
import subprocess
import soundfile as sf
import shutil 

def unify_sample_rate_and_bitrate(input_audio_path, target_sr=22050, target_bitrate='128k'):
    
    temp_audio_path = "temp_audio.wav"

    command = [
        'ffmpeg',
        '-i', input_audio_path,
        '-ar', str(target_sr),
        '-b:a', target_bitrate,
        temp_audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    y, sr = librosa.load(temp_audio_path, sr=target_sr)

    os.remove(temp_audio_path)

    return y, sr


def separate_vocals(y, sr, model='spleeter:2stems'):
  
    temp_audio_path = "temp_vocals_input.wav"
    sf.write(temp_audio_path, y, sr)

    separator = Separator(model)

    separator.separate_to_file(temp_audio_path, "output")
    os.remove(temp_audio_path)

    vocals_path = os.path.join("output", "temp_vocals_input", "vocals.wav")
    vocals, _ = librosa.load(vocals_path, sr=sr)

    shutil.rmtree(os.path.join("output", "temp_vocals_input"))  # 替换 os.rmdir
    shutil.rmtree("output")  # 替换 os.rmdir

    return vocals

def trim_silence(y, sr, top_db=20, frame_length=2048, hop_length=512):
 
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    
    return y_trimmed

def random_slices(y, sr, num_slices=3, slice_length_ratio=0.1):

    total_duration = len(y) / sr
    slice_duration = total_duration * slice_length_ratio
    slice_samples = int(slice_duration * sr)

    if slice_samples > len(y):
        raise ValueError("音频长度不足以生成所需的切片数量。")

    slices = []
    
    for _ in range(num_slices):
        start_sample = np.random.randint(0, len(y) - slice_samples)
        slice_y = y[start_sample:start_sample + slice_samples]
    
        slices.append(slice_y)
    return slices

def compute_mel_spectrogram(slices, output_path, sr=22050, n_mels=128):

    mel_spectrograms = []
    for slice_y in slices:
        mel_spectrogram = librosa.feature.melspectrogram(y=slice_y, sr=sr, n_mels=n_mels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrograms.append(mel_spectrogram)
    mel_spectrograms_array = np.array(mel_spectrograms)
    np.save(output_path, mel_spectrograms_array)
    print(f"梅尔频谱数组保存到 {output_path}")

def load_mel_spectrogram_array(file_path):

    mel_spectrograms_array = np.load(file_path)
    shape = mel_spectrograms_array.shape
    print(f"加载的梅尔频谱数组形状: {shape}")
    print(f"梅尔频谱数组内容: {mel_spectrograms_array}") 
    
    return mel_spectrograms_array, shape

if __name__ == "__main__":
    input_audio_path = "Sample.mp3"  
    output_path = "data.npy"  
    y, sr = unify_sample_rate_and_bitrate(input_audio_path)
    vocals = separate_vocals(y, sr)
    vocals_trimmed = trim_silence(vocals, sr)
    slices = random_slices(vocals_trimmed, sr)
    compute_mel_spectrogram(slices, output_path)
    mel_spectrograms_array, shape = load_mel_spectrogram_array(output_path)