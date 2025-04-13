import os
from spleeter.separator import Separator
import tensorflow as tf

def separate_vocals_from_folder(input_folder, output_folder, model='spleeter:2stems', start_index=1, end_index=100):
    os.makedirs(output_folder, exist_ok=True)
    separator = Separator(model)
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".mp3")])

    for filename in files[start_index-1:end_index]:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0])
        separator.separate_to_file(input_path, output_path)

        print(f"提取完成：{filename} -> {output_path}/vocals.wav")

    print("当前批次处理完成！")

# 示例：从 'songs' 文件夹中提取 1 到 100 的人声，保存到 'vocals' 文件夹中
if __name__ == "__main__": 
    input_folder = r"Musics/SP"  
    output_folder = r"Separate/Vocal/SP"  

    separate_vocals_from_folder(input_folder, output_folder, start_index=1,end_index=400)
                                                                                                      