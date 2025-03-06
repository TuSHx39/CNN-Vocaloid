import os
import librosa
import numpy as np

# 提取 MFCC 特征
def extract_mfcc(audio_path, sr=22050, n_mfcc=13):
    """
    提取 MFCC 特征。
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = librosa.power_to_db(mfccs, ref=np.max)
    return mfccs

# 数据归一化
def normalize_mfcc(mfccs):
    """
    对 MFCC 特征进行归一化处理。
    """
    mfccs = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs))
    return mfccs

# 处理音频文件并保存特征
def process_and_save_vocals(input_folder, output_folder, start_index=1, end_index=100):
    """
    处理指定范围内的音频文件，提取 MFCC 特征，并将特征保存为 NumPy 文件。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有文件
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".wav")])

    # 遍历指定范围内的文件
    for filename in files[start_index-1:end_index]:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_mfcc.npy")

        # 提取 MFCC 特征
        mfccs = extract_mfcc(input_path)
        mfccs = normalize_mfcc(mfccs)

        # 保存特征为 NumPy 文件
        np.save(output_path, mfccs)
        print(f"处理完成：{filename} -> {output_path}")

# 示例：处理并保存特征
if __name__ == "__main__":
    input_folder = r"C:\path\to\your\vocals"  # 替换为你的分离后的人声文件夹路径
    output_folder = r"C:\path\to\your\features"  # 替换为你希望保存特征的文件夹路径

    # 处理并保存特征
    process_and_save_vocals(input_folder, output_folder, start_index=1, end_index=100)