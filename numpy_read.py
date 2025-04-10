import numpy as np

def load_mel_spectrogram_array(file_path):
    mel_spectrograms_array = np.load(file_path)
    shape = mel_spectrograms_array.shape
    print(f"加载的梅尔频谱数组形状: {shape}")
    print(f"梅尔频谱数组内容: {mel_spectrograms_array}") 
    return mel_spectrograms_array, shape

mel_spectrograms_array, shape = load_mel_spectrogram_array("data\\1\\data.npy")