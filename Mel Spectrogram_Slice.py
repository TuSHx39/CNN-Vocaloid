import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

def random_slices(audio_path, num_slices=2, slice_length_ratio=0.1, sr=44100, n_mels=128, fmax=None):
    """
    随机切片音频并计算每段切片的梅尔频谱图。

    参数:
        audio_path (str): 音频文件路径。
        num_slices (int): 切片数量，默认为2。
        slice_length_ratio (float): 切片长度占音频总长度的比例，默认为0.1。
        sr (int): 采样率，默认为44100Hz。
        n_mels (int): 梅尔频谱图的频率维度，默认为128。
        fmax (int): 最大频率，默认为采样率的一半（奈奎斯特频率）。
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr  # 音频总时长（秒）
    slice_duration = duration * slice_length_ratio  # 切片时长（秒）
    slice_samples = int(slice_duration * sr)  # 切片的样本数

    # 随机选择切片的起始点
    start_points = random.sample(range(len(y) - slice_samples), num_slices)

    # 计算每段切片的梅尔频谱图
    for i, start in enumerate(start_points):
        # 提取切片
        slice_y = y[start:start + slice_samples]

        # 计算梅尔频谱图
        mel_spectrogram = librosa.feature.melspectrogram(
            y=slice_y, sr=sr, n_mels=n_mels, fmax=fmax
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # 可视化梅尔频谱图
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', fmax=fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram (Slice {i+1})')
        plt.tight_layout()

        # 保存图像
        output_path = f"mel_spectrogram_slice_{i+1}.png"
        plt.savefig(output_path)
        print(f"梅尔频谱图（切片 {i+1}）已保存为 {output_path}")

        # 关闭图像窗口
        plt.close()

# 示例：随机切片并计算梅尔频谱图
if __name__ == "__main__":
    # 替换为你的音频文件路径
    audio_path = "Separate/Sample/vocals.wav"

    # 调整参数（可选）
    num_slices = 2  # 切片数量
    slice_length_ratio = 0.1  # 切片长度占音频总长度的比例
    sr = 44100  # 采样率
    n_mels = 128  # 梅尔频谱图的频率维度
    fmax = 8000  # 最大频率

    # 随机切片并计算梅尔频谱图
    random_slices(audio_path, num_slices=num_slices, slice_length_ratio=slice_length_ratio, sr=sr, n_mels=n_mels, fmax=fmax)