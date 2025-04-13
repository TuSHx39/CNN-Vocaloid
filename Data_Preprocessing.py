import os
import librosa
import numpy as np
import soundfile as sf

def trim_silence(y, sr, top_db=8, frame_length=2048, hop_length=512):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return y_trimmed

def random_slices(y, sr, num_slices=3, slice_duration=10):
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
    mel_spectrograms = []
    for slice_y in slices:
        mel_spectrogram = librosa.feature.melspectrogram(y=slice_y, sr=sr, n_mels=n_mels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrograms.append(mel_spectrogram)
    
    mel_spectrograms_array = np.array(mel_spectrograms, dtype='float32')
    mel_spectrograms_array = (mel_spectrograms_array - np.min(mel_spectrograms_array)) / (np.max(mel_spectrograms_array) - np.min(mel_spectrograms_array))
    np.save(output_path, mel_spectrograms_array)


def batch_process_audio(input_dir, output_dir, target_sr=44100, target_bitrate='128k'):
    os.makedirs(output_dir, exist_ok=True)

    for folder in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            output_subdir = os.path.join(output_dir, folder)
            os.makedirs(output_subdir, exist_ok=True)

            for audio_file in os.listdir(folder_path):
                if audio_file.endswith(".wav"):
                    input_audio_path = os.path.join(folder_path, audio_file)
                    output_path = os.path.join(output_subdir, f"{os.path.splitext(audio_file)[0]}.npy")

                    # 加载音频文件
                    y, sr = librosa.load(input_audio_path, sr=target_sr)

                    # 去除静音部分
                    vocals_trimmed = trim_silence(y, sr)

                    # 随机切片
                    slices = random_slices(vocals_trimmed, sr)

                    # 计算梅尔频谱并保存为 NumPy 数组
                    compute_mel_spectrogram(slices, output_path, sr)

                    print(f"文件 {audio_file} 处理结果已保存至 {output_path}")

                    # 释放内存
                    del y, vocals_trimmed, slices

if __name__ == "__main__":
    input_dir = "C:\Data\Vocal"  # 替换为包含音频文件夹的路径
    output_dir = "data"  # 替换为保存梅尔频谱数组的路径
    batch_process_audio(input_dir, output_dir)