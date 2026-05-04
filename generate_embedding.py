import requests
import json
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor



def embed(text_list, base_url):
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

if __name__ == "__main__":
    # 服务器地址
    base_urls = [
        "http://localhost:5000",
        "http://localhost:5001",
        "http://localhost:5002", 
        "http://localhost:5003",
        "http://localhost:5005"
    ]
    
    # 获取patent_data文件夹下的所有csv文件的文件路径
    import os
    import glob
    
    # 指定patent_data文件夹路径
    patent_data_folder = "patent_data"
    patent_embedding_folder = "patent_embedding"
    
    # 创建嵌入向量保存文件夹（如果不存在）
    os.makedirs(patent_embedding_folder, exist_ok=True)
    
    # 获取所有csv文件路径
    csv_files = glob.glob(os.path.join(patent_data_folder, "*.csv"))
    
    IPC_3s = []
    for file in csv_files:
        # 从文件名中提取IPC_3代码
        file_name = os.path.basename(file)
        if "patent_data_" in file_name and "_cleaned.csv" in file_name:
            ipc_code = file_name.replace("patent_data_", "").replace("_cleaned.csv", "")
            IPC_3s.append(ipc_code)
    
    # 定义处理任务函数
    def process_file(ipc_code, base_url):
        try:
            # 构建文件路径
            patent_file = f'{patent_data_folder}/patent_data_{ipc_code}_cleaned.csv'
            
            # 读取专利数据
            patent_df = pd.read_csv(patent_file)
            patent_df = patent_df.fillna("")
            
            print(f"开始处理 {ipc_code} 的标题，使用服务 {base_url}")
            
            # 处理专利标题
            field = 'title'
            id_field = 'id'
            batch_size = 5000
            output_file = f'{patent_embedding_folder}/patent_title_{ipc_code}_embeddings.npz'
            process_embeddings(patent_df, field, id_field, batch_size, output_file, base_url=base_url, save_batch_size=1000)
            
            print(f"开始处理 {ipc_code} 的摘要，使用服务 {base_url}")
            
            # 处理专利摘要
            field = 'brief'
            batch_size = 300
            output_file = f'{patent_embedding_folder}/patent_brief_{ipc_code}_embeddings.npz'
            process_embeddings(patent_df, field, id_field, batch_size, output_file, base_url=base_url, save_batch_size=3000)
            
            print(f"完成处理 {ipc_code}，使用服务 {base_url}")
            return True
        except Exception as e:
            print(f"处理 {ipc_code} 时出错: {str(e)}")
            return False
    
    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=len(base_urls)) as executor:
        # 初始提交任务，每个服务分配一个文件
        future_to_ipc = {}
        for i, ipc_code in enumerate(IPC_3s[:min(len(base_urls), len(IPC_3s))]):
            future = executor.submit(process_file, ipc_code, base_urls[i])
            future_to_ipc[future] = (ipc_code, base_urls[i], i)
        
        # 待处理的IPC代码索引
        next_ipc_index = min(len(base_urls), len(IPC_3s))
        
        # 当一个任务完成时，提交新任务
        import concurrent.futures
        while future_to_ipc:
            # 获取已完成的任务
            done, _ = concurrent.futures.wait(
                future_to_ipc, 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                ipc_code, base_url, url_index = future_to_ipc[future]
                del future_to_ipc[future]
                
                try:
                    result = future.result()
                except Exception as e:
                    print(f"处理 {ipc_code} 时发生异常: {str(e)}")
                
                # 如果还有待处理的文件，则分配给空闲的服务
                if next_ipc_index < len(IPC_3s):
                    next_ipc = IPC_3s[next_ipc_index]
                    future = executor.submit(process_file, next_ipc, base_url)
                    future_to_ipc[future] = (next_ipc, base_url, url_index)
                    next_ipc_index += 1
    
    print("所有文件处理完成！")