import os
import librosa
import numpy as np
from spleeter.separator import Separator
import subprocess
import soundfile as sf
import shutil 

# 统一采样率和码率
def unify_sample_rate_and_bitrate(input_audio_path, target_sr=22050, target_bitrate='128k'):
    
    # 临时文件路径
    temp_audio_path = "temp_audio.wav"

    command = [
        'ffmpeg',
        '-i', input_audio_path,
        '-ar', str(target_sr),
        '-b:a', target_bitrate,
        temp_audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 加载处理后的音频文件
    y, sr = librosa.load(temp_audio_path, sr=target_sr)

    # 删除临时文件
    os.remove(temp_audio_path)

    return y, sr

# 第二步：分离人声
def separate_vocals(y, sr, model='spleeter:2stems'):
    """
    使用 Spleeter 分离音频中的人声部分。
    """
    # 保存音频到临时文件
    temp_audio_path = "temp_vocals_input.wav"
    sf.write(temp_audio_path, y, sr)

    # 初始化 Spleeter 分离器
    separator = Separator(model)

    # 分离音频并返回人声部分
    separator.separate_to_file(temp_audio_path, "output")
    os.remove(temp_audio_path)

    # 加载分离后的人声部分
    vocals_path = os.path.join("output", "temp_vocals_input", "vocals.wav")
    vocals, _ = librosa.load(vocals_path, sr=sr)

    # 删除分离结果文件
    shutil.rmtree(os.path.join("output", "temp_vocals_input"))  # 替换 os.rmdir
    shutil.rmtree("output")  # 替换 os.rmdir

    return vocals

# 第三步：去除静音部分
def trim_silence(y, sr, top_db=20, frame_length=2048, hop_length=512):
    """
    去除音频中的静音部分。
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return y_trimmed

# 第四步：随机切片
def random_slices(y, sr, num_slices=3, slice_length_ratio=0.1):
    """
    从音频中随机切片。
    """
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

# 第五步：计算梅尔频谱并保存为 NumPy 数组
def compute_mel_spectrogram(slices, output_path, sr=22050, n_mels=128):
    """
    计算音频切片的梅尔频谱并保存为 NumPy 数组。
    """
    mel_spectrograms = []
    for slice_y in slices:
        mel_spectrogram = librosa.feature.melspectrogram(y=slice_y, sr=sr, n_mels=n_mels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrograms.append(mel_spectrogram)
    
    # 将所有梅尔频谱图堆叠成一个 NumPy 数组
    mel_spectrograms_array = np.array(mel_spectrograms)
    np.save(output_path, mel_spectrograms_array)
    print(f"梅尔频谱数组保存到 {output_path}")

# 主函数
if __name__ == "__main__":
    # 输入文件路径
    input_audio_path = "Sample.mp3"  # 替换为你的音频文件路径

    # 输出文件路径
    output_path = "data.npy"  # 替换为你希望保存 NumPy 数组的路径

    # 第一步：统一采样率和码率
    y, sr = unify_sample_rate_and_bitrate(input_audio_path)

    # 第二步：分离人声
    vocals = separate_vocals(y, sr)

    # 第三步：去除静音部分
    vocals_trimmed = trim_silence(vocals, sr)

    # 第四步：随机切片
    slices = random_slices(vocals_trimmed, sr)

    # 第五步：计算梅尔频谱并保存为 NumPy 数组
    compute_mel_spectrogram(slices, output_path)