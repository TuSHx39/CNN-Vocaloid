import os
import numpy as np

def merge_arrays_in_folder(input_folder, output_folder):

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有子文件夹
    subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        output_path = os.path.join(output_folder, f"{subfolder}.npy")

        # 获取子文件夹中的所有 .npy 文件
        npy_files = [f for f in os.listdir(subfolder_path) if f.endswith(".npy")]

        # 加载并合并数组
        merged_array = None
        for npy_file in npy_files:
            file_path = os.path.join(subfolder_path, npy_file)
            array = np.load(file_path)
            if merged_array is None:
                merged_array = array
            else:
                merged_array = np.concatenate((merged_array, array), axis=0)

        # 保存合并后的数组
        np.save(output_path, merged_array)
        print(f"合并后的数组已保存到 {output_path}")

if __name__ == "__main__":
  
    input_folder = "data"  

    output_folder = "Training Set" 


    merge_arrays_in_folder(input_folder, output_folder)