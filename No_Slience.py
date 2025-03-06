import librosa
import numpy as np
import os

def trim_silence(audio_path, output_path, top_db=20, frame_length=2048, hop_length=512, start_index=1, end_index=100):
    """
    去除音频中的静音部分，并保存处理后的音频。

    参数:
        audio_path (str): 输入音频文件的路径。
        output_path (str): 处理后音频文件的保存路径。
        top_db (int): 静音的阈值，单位为分贝。
        frame_length (int): 每帧的样本数。
        hop_length (int): 帧之间的样本数。
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=None)

    # 去除静音部分
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    # 保存处理后的音频
    librosa.output.write_wav(output_path, y_trimmed, sr)
    print(f"处理完成：{audio_path} -> {output_path}")

# 示例：去除音频中的静音部分
if __name__ == "__main__":
    input_audio_path = "path/to/your/audio/file.wav"  # 替换为你的音频文件路径
    output_audio_path = "path/to/your/output/file_trimmed.wav"  # 替换为你希望保存处理后音频的路径

    # 去除静音部分
    trim_silence(input_audio_path, output_audio_path)