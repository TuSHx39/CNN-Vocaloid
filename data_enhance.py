import os
import librosa
import numpy as np
import soundfile as sf

def process_audio_files(input_folder, output_folder, min_pitch_shift=-2, max_pitch_shift=2, min_speed_change=0.8, max_speed_change=1.2):
    """
    批量处理音频文件，随机加速、减速或升降调。

    参数:
        input_folder (str): 包含音频文件的输入文件夹路径。
        output_folder (str): 保存处理后的音频文件的输出文件夹路径。
        min_pitch_shift (float): 最小升降调范围（单位为半音）。
        max_pitch_shift (float): 最大升降调范围（单位为半音）。
        min_speed_change (float): 最小速度变化范围（0.8 表示减速 20%，1.2 表示加速 20%）。
        max_speed_change (float): 最大速度变化范围。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有文件
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".mp3")])

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'{filename}_enhance.mp3')

        # 加载音频文件
        y, sr = librosa.load(input_path, sr=None)

        # 随机生成升降调和速度变化参数
        pitch_shift = np.random.uniform(min_pitch_shift, max_pitch_shift)
        speed_change = np.random.uniform(min_speed_change, max_speed_change)

        # 应用升降调
        y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_shift)

        # 应用速度变化
        y_stretched = librosa.effects.time_stretch(y_shifted, rate=speed_change)

        # 保存处理后的音频文件
        sf.write(output_path, y_stretched, sr)
        print(f"处理完成：{filename} -> {output_path}")

# 示例：处理 'songs' 文件夹中的所有音频文件，保存到 'processed_songs' 文件夹中
if __name__ == "__main__":
    input_folder = r"./enhance"  # 替换为你的音频文件夹路径
    output_folder = r"./enhance/enhance"  # 替换为保存处理后音频的文件夹路径

    process_audio_files(input_folder, output_folder)