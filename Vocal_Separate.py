import os
from spleeter.separator import Separator

def separate_vocals_from_folder(input_folder, output_folder, model='spleeter:2stems', start_index=1, end_index=100):
    """
    从指定文件夹中提取指定范围内的歌曲的人声部分，并将人声保存到输出文件夹中。

    参数:
        input_folder (str): 包含音频文件的输入文件夹路径。
        output_folder (str): 保存提取的人声的输出文件夹路径。
        model (str): 使用的 Spleeter 模型，默认为 'spleeter:2stems'（人声 + 伴奏）。
        start_index (int): 开始处理的文件索引（包含）。
        end_index (int): 结束处理的文件索引（包含）。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 初始化 Spleeter 分离器
    separator = Separator(model)

    # 获取输入文件夹中的所有文件
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".mp3")])

    # 遍历指定范围内的文件
    for filename in files[start_index-1:end_index]:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0])

        # 分离音频并保存人声部分
        separator.separate_to_file(input_path, output_path)

        print(f"提取完成：{filename} -> {output_path}/vocals.wav")

    print("当前批次处理完成！")

# 示例：从 'songs' 文件夹中提取 1 到 100 的人声，保存到 'vocals' 文件夹中
if __name__ == "__main__":
    input_folder = r"Musics\Miku"  
    output_folder = r"Separate\Vocal\Miku"  

    separate_vocals_from_folder(input_folder, output_folder, start_index=201,end_index=445)