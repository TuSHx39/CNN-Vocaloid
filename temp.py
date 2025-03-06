import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compute_and_display_mel_spectrogram(audio_path, sr=44100, n_mels=128, fmax=None):
    """
    计算并显示梅尔频谱图（Mel-Spectrogram）。

    参数:
        audio_path (str): 音频文件路径。
        sr (int): 采样率，默认为44100Hz。
        n_mels (int): 梅尔频谱图的频率维度，默认为128。
        fmax (int): 最大频率，默认为采样率的一半（奈奎斯特频率）。
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 计算梅尔频谱图
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmax=fmax
    )
    
    # 将功率谱转换为对数尺度（分贝）
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    # 可视化梅尔频谱图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', fmax=fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()

    # 保存梅尔频谱图到文件
    plt.savefig("mel_spectrogram_no_slience.png")
    print("梅尔频谱图已保存为 'mel_spectrogram_no_slience.png'。")

    # 显示梅尔频谱图
    plt.show()

    # 返回对数梅尔频谱图
    return log_mel_spectrogram

# 示例：计算并显示梅尔频谱图
if __name__ == "__main__":
    audio_path = "No Slience\Sample\sample_no_slience.wav"  
    # 调整参数（可选）
    sr = 44100  # 采样率
    n_mels = 128  # 梅尔频谱图的频率维度
    fmax = 8000  # 最大频率

    # 计算并显示梅尔频谱图
    log_mel_spectrogram = compute_and_display_mel_spectrogram(audio_path, sr=sr, n_mels=n_mels, fmax=fmax)