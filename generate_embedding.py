import requests
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from config import (
    EMBEDDING_SERVERS,
    EMBEDDING_BATCH_SIZES,
    EMBEDDING_SAVE_BATCH_SIZE,
    PATENT_DATA_DIR,
    PATENT_EMBEDDING_DIR,
)



def embed(text_list, base_url):
    url = f"{base_url}/embed"

    payload = {
        "texts": text_list
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
    response.raise_for_status()
    result = response.json()
    if "embeddings" not in result:
        raise ValueError("embedding 服务响应缺少 embeddings 字段")
    return result["embeddings"]

def process_embeddings(patent_df, field, id_field, batch_size, output_file, base_url, save_batch_size=1000):
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
        embeddings = embed(batch_texts, base_url=base_url)
        if len(embeddings) != len(batch_ids):
            raise ValueError(f"embedding 数量与输入数量不一致: {len(embeddings)} != {len(batch_ids)}")

        current_embeddings.extend(embeddings)
        current_ids.extend(batch_ids)
        record_count += len(batch_ids)

        if record_count >= save_batch_size*batch_size:
            save_batch(current_embeddings, current_ids, id_field, f"{file_name}_{batch_count}{file_extension}")
            print(f"已保存批次 {batch_count}，包含 {len(current_ids)} 条记录")

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
    if not output_file.endswith('.npz'):
        output_file = os.path.splitext(output_file)[0] + '.npz'

    embeddings_array = np.array(embeddings)
    ids_array = np.array(ids)

    np.savez_compressed(
        output_file,
        embeddings=embeddings_array,
        ids=ids_array
    )

    print(f"数据已保存到 {output_file}")


def process_file(ipc_code, base_url,
                 patent_data_dir=None, patent_embedding_dir=None,
                 batch_sizes=None, save_batch_size=None):
    """处理单个 IPC 代码的 title 和 brief 向量化。"""
    _patent_data_dir = patent_data_dir or PATENT_DATA_DIR
    _patent_embedding_dir = patent_embedding_dir or PATENT_EMBEDDING_DIR
    _batch_sizes = batch_sizes or EMBEDDING_BATCH_SIZES
    _save_batch_size = save_batch_size or EMBEDDING_SAVE_BATCH_SIZE

    patent_file = os.path.join(_patent_data_dir, f'patent_data_{ipc_code}_cleaned.csv')
    patent_df = pd.read_csv(patent_file).fillna("")

    for field, batch_size in _batch_sizes.items():
        print(f"开始处理 {ipc_code} 的 {field}，使用服务 {base_url}")
        output_file = os.path.join(_patent_embedding_dir, f'patent_{field}_{ipc_code}_embeddings.npz')
        process_embeddings(patent_df, field, 'id', batch_size, output_file,
                           base_url=base_url, save_batch_size=_save_batch_size)

    print(f"完成处理 {ipc_code}")


def main():
    os.makedirs(PATENT_EMBEDDING_DIR, exist_ok=True)

    ipc_codes = [
        os.path.basename(f).replace("patent_data_", "").replace("_cleaned.csv", "")
        for f in glob.glob(os.path.join(PATENT_DATA_DIR, "patent_data_*_cleaned.csv"))
    ]

    failures = []
    with ThreadPoolExecutor(max_workers=len(EMBEDDING_SERVERS)) as executor:
        future_to_ipc = {
            executor.submit(process_file, ipc_code, EMBEDDING_SERVERS[i]): (ipc_code, EMBEDDING_SERVERS[i], i)
            for i, ipc_code in enumerate(ipc_codes[:len(EMBEDDING_SERVERS)])
        }
        next_ipc_index = len(future_to_ipc)

        while future_to_ipc:
            done, _ = concurrent.futures.wait(
                future_to_ipc,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                ipc_code, base_url, url_index = future_to_ipc.pop(future)
                try:
                    future.result()
                except Exception as e:
                    failures.append((ipc_code, e))
                    print(f"处理 {ipc_code} 时发生异常: {e}")
                else:
                    if next_ipc_index < len(ipc_codes):
                        next_ipc = ipc_codes[next_ipc_index]
                        future_to_ipc[executor.submit(process_file, next_ipc, base_url)] = (next_ipc, base_url, url_index)
                        next_ipc_index += 1

    if failures:
        failed_ipcs = ', '.join(ipc_code for ipc_code, _ in failures)
        raise RuntimeError(f"以下 IPC 处理失败: {failed_ipcs}")

    print("所有文件处理完成！")


if __name__ == "__main__":
    main()
