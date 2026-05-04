import pandas as pd
import numpy as np
import os
import glob

def convert_csv_to_npz(csv_file_path):
    """将CSV格式的嵌入向量文件转换为NPZ格式"""
    npz_file_path = 'patent_brief_G06F_embeddings_0.npz'
    
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 提取ID列
    id_field = 'id'  # 假设ID列名为'id'，根据需要修改
    ids = df[id_field].values
    
    # 提取嵌入向量列
    embedding_cols = [col for col in df.columns if col.startswith('dim_')]
    embeddings = df[embedding_cols].values
    
    # 保存为NPZ文件
    np.savez_compressed(
        npz_file_path,
        embeddings=embeddings,
        ids=ids
    )
    
    print(f"转换完成: {csv_file_path} -> {npz_file_path}")
    return npz_file_path

def batch_convert_csv_to_npz(csv_file):
    """批量转换文件夹中的所有CSV嵌入文件为NPZ格式"""
    npz_file = convert_csv_to_npz(csv_file)

    return npz_file

# 使用示例
if __name__ == "__main__":
    converted_files = batch_convert_csv_to_npz("patent_brief_G06F_embeddings_0.csv")