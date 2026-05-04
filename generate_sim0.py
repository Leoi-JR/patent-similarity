import requests
import json
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
import os

# 服务器地址
base_url = "http://localhost:5000"


def embed(text_list):
    url = f"{base_url}/embed"
    
    # 准备要嵌入的文本列表
    payload = {
        "texts": text_list
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # 发送POST请求
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # 返回结果
    if response.status_code == 200:
        result = response.json()
        return result["embeddings"]
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
        return None

def process_embeddings(patent_df, field, id_field, batch_size, output_file, save_batch_size=1000):
    # 获取总数据量
    total_records = len(patent_df)
    # 拆分文件名和后缀
    file_name, file_extension = os.path.splitext(output_file)
    
    # 初始化批次计数和记录计数
    batch_count = 0
    record_count = 0
    
    # 初始化当前批次的嵌入和ID列表
    current_embeddings = []
    current_ids = []
    
    # 使用tqdm创建进度条
    for i in tqdm(range(0, total_records, batch_size), desc="处理数据批次"):
        # 获取当前批次的数据
        batch_df = patent_df.iloc[i:min(i+batch_size, total_records)]
        # 获取文本和ID
        batch_texts = batch_df[field].values.tolist()
        batch_ids = batch_df[id_field].values.tolist()
        
        # 获取嵌入向量
        embeddings = embed(batch_texts)
        if embeddings:
            current_embeddings.extend(embeddings)
            current_ids.extend(batch_ids)
            record_count += len(batch_ids)
            
            # 检查是否需要保存当前批次
            if record_count >= save_batch_size*batch_size:
                # 保存当前批次
                save_batch(current_embeddings, current_ids, id_field, f"{file_name}_{batch_count}{file_extension}")
                print(f"已保存批次 {batch_count}，包含 {len(current_ids)} 条记录")
                
                # 重置当前批次的列表
                current_embeddings = []
                current_ids = []
                batch_count += 1
                record_count = 0
    
    # 保存剩余的记录
    if current_ids:
        save_batch(current_embeddings, current_ids, id_field, f"{file_name}_{batch_count}{file_extension}")
        print(f"已保存最后批次 {batch_count}，包含 {len(current_ids)} 条记录")

def save_batch(embeddings, ids, id_field, output_file):
    """保存一个批次的嵌入向量到NPZ文件"""
    # 将文件扩展名改为.npz
    if not output_file.endswith('.npz'):
        output_file = os.path.splitext(output_file)[0] + '.npz'
    
    # 将嵌入向量转换为numpy数组
    embeddings_array = np.array(embeddings)
    ids_array = np.array(ids)
    
    # 保存到NPZ文件
    np.savez_compressed(
        output_file,
        embeddings=embeddings_array,
        ids=ids_array
    )
    
    print(f"数据已保存到 {output_file}")

BATCH_SIZES = {
    'title': 5000,
    'brief': 300,
}

if __name__ == "__main__":
    IPC_3 = "G06F"
    field = 'title'
    id_field = 'id'
    patent_file = f'patent_data_{IPC_3}_cleaned.csv'
    batch_size = BATCH_SIZES[field]
    output_file = f'patent_{field}_{IPC_3}_embeddings.npz'

    # 读取专利数据
    patent_df = pd.read_csv(patent_file)
    patent_df = patent_df.fillna("")

    # 处理所有数据
    process_embeddings(patent_df, field, id_field, batch_size, output_file)