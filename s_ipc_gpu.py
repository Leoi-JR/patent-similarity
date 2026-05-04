import pandas as pd
import numpy as np
import os
import gc
import time
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import csr_matrix, save_npz, vstack, load_npz
from tqdm import tqdm
def setup_gpu(gpu_id=0):
    """设置并初始化GPU"""
    cp.cuda.Device(gpu_id).use()
    device_name = cp.cuda.runtime.getDeviceProperties(gpu_id)['name'].decode()
    print(f"使用GPU {gpu_id}: {device_name}")

def load_data(patent_file, ipc_file):
    """加载和预处理专利数据"""
    print("正在读取数据文件...")
    patent_df = pd.read_csv(patent_file)
    ipc_df = pd.read_csv(ipc_file)
    
    print("正在合并IPC类别信息...")
    merged_df = pd.merge(
        patent_df,
        ipc_df,
        left_on='main_ipc',
        right_on='level_code',
        how='inner'
    )
    
    # 只保留需要的列以减少内存使用
    needed_columns = ['id', 'title', 'brief', 'level1_code', 'level2_code', 
                       'level3_code', 'level4_code', 'level5_code']
    return merged_df[needed_columns]

def create_feature_matrices(merged_df, weights):
    """为每个IPC级别创建特征矩阵"""
    print("进行高效预处理...")
    n = len(merged_df)
    feature_matrices = {}
    feature_matrices_gpu = {}
    
    for level, weight in weights.items():
        print(f"处理 {level}...")
        # 创建编码字典
        unique_values = merged_df[level].dropna().unique()
        code_to_idx = {code: idx for idx, code in enumerate(unique_values)}
        
        # 为每个专利创建行索引和列索引
        rows, cols, data = [], [], []
        
        for i, code in enumerate(merged_df[level]):
            if pd.notna(code):
                rows.append(i)
                cols.append(code_to_idx[code])
                data.append(1.0)  # 明确使用float类型
        
        # 创建稀疏矩阵，明确指定数据类型为float32
        feature_matrices[level] = csr_matrix((np.array(data, dtype=np.float32), 
                                             (np.array(rows), np.array(cols))), 
                                             shape=(n, len(code_to_idx)))
        
        # 转换为GPU稀疏矩阵
        feature_matrices_gpu[level] = cp_csr_matrix(feature_matrices[level])
        
        # # 保存编码字典以便后续使用
        # pd.DataFrame({'code': list(code_to_idx.keys()), 
        #               'index': list(code_to_idx.values())}).to_csv(
        #     f'{output_dir}/{level}_encoding.csv', index=False)
        
        # 清理内存
        del rows, cols, data
        gc.collect()
    
    return feature_matrices_gpu, n

def process_batch_gpu(i, batch_size, n, feature_matrices_gpu, weights, brief_embeddings_array, title_embeddings_array, patent_ids):
    """处理单个批次的相似度计算并返回每行中值大于0.85的列索引及对应相似度值"""
    start_i = i * batch_size
    end_i = min((i + 1) * batch_size, n)
    
    # 创建一个组合的相似度矩阵
    combined_similarity = None
    
    for level, weight in weights.items():
        # 获取当前批次的特征矩阵（GPU版本）
        batch_features = feature_matrices_gpu[level][start_i:end_i]
        
        # 计算当前批次与所有专利的相似度（GPU加速）
        level_similarity = batch_features.dot(feature_matrices_gpu[level].T)
        
        # 应用权重
        level_similarity = level_similarity.multiply(weight)
        
        # 合并到总相似度矩阵
        if combined_similarity is None:
            combined_similarity = level_similarity
        else:
            combined_similarity = combined_similarity + level_similarity
    
    # 对角线设为1（专利与自身的相似度）- 使用高效的矩阵操作替代循环
    diagonal_indices = cp.arange(start_i, end_i)
    row_indices = cp.arange(end_i - start_i)
    combined_similarity[row_indices, diagonal_indices] = 1.0

    # 计算摘要的embedding相似度
    brief_similarity = cp.dot(brief_embeddings_array[start_i:end_i], brief_embeddings_array.T)
    title_similarity = cp.dot(title_embeddings_array[start_i:end_i], title_embeddings_array.T)
    
    # 将combined_similarity从稀疏矩阵转换为密集矩阵，然后再相加
    combined_similarity = combined_similarity.toarray()*0.4 + brief_similarity*0.4 + title_similarity*0.2
    
    # 在GPU上识别大于阈值的元素
    # similarity_mask = (combined_similarity > 0.75)
    
    # 将相似度矩阵和掩码转移到CPU上处理
    similarity_cpu = combined_similarity.get()
    # similarity_mask_cpu = similarity_mask.get()
    
    # 创建结果列表，同时包含索引和值
    high_similarity_results = []
    for row_idx in range(similarity_cpu.shape[0]):
        # 获取当前行所有相似度值
        row_similarities = similarity_cpu[row_idx]
        
        # 使用部分排序找出前1000个最大值的索引
        # 如果大于阈值的结果少于1000个，则获取所有大于阈值的结果
        mask = row_similarities > 0.75
        if np.sum(mask) > 1000:
            # 找出大于阈值的值的索引
            valid_indices = np.where(mask)[0]
            # 对这些索引对应的值进行部分排序，找出前1000大的索引
            top_k_indices = valid_indices[np.argsort(row_similarities[valid_indices])[-1000:]]
            # 按相似度降序排列
            top_k_indices = top_k_indices[np.argsort(-row_similarities[top_k_indices])]
        else:
            # 如果大于阈值的结果少于1000个，直接获取所有大于阈值的结果
            valid_indices = np.where(mask)[0]
            # 按相似度降序排列
            top_k_indices = valid_indices[np.argsort(-row_similarities[valid_indices])]
        
        # 获取这些位置的相似度值
        top_values = row_similarities[top_k_indices]
        
        # 将索引和值打包在一起
        row_results = [(patent_ids[int(row_idx+start_i)], patent_ids[int(col_idx)], float(top_values[i])) 
                      for i, col_idx in enumerate(top_k_indices)]
        
        high_similarity_results.append(row_results)
    
    return high_similarity_results

def merge_similarity_matrices(total_batches, output_dir):
    """合并所有批次的相似度矩阵"""
    print("合并相似度矩阵...")
    combined = None
    for i in range(total_batches):
        batch_sim = load_npz(f'{output_dir}/similarity_batch_{i}.npz')
        if combined is None:
            combined = batch_sim
        else:
            combined = vstack([combined, batch_sim])
    save_npz(f'{output_dir}/complete_similarity_matrix.npz', combined)
    print("合并完成，保存为 complete_similarity_matrix.npz")

# 加载专利的embedding npz
def load_patent_embeddings(file_name):
    data = cp.load(file_name)
    embeddings_array = data['embeddings']
    ids_array = data['ids']
    return embeddings_array, ids_array

def save_similiarity_results(similiarity_results, output_dir, ipc_code):
    # 将结果保存为parquet文件，更高效
    df = pd.DataFrame(similiarity_results, columns=['patent_id', 'similar_patent_id', 'similarity_score'])
    df.to_parquet(f'{output_dir}/similiarity_results_{ipc_code}.parquet', index=False)

def main():
    import os
    import glob

    # 设置GPU
    gpu_id = 3  # 修改这个值来选择不同的GPU
    setup_gpu(gpu_id)

    # 定义各层级权重
    weights = {
        'level3_code': 0.2,
        'level4_code': 0.3,
        'level5_code': 0.5
    }

    patent_embedding_folder = "patent_embedding"
    patent_data_folder = "patent_data"
    output_dir = 'similarity_results_gpu'
    # 获取所有npz文件路径
    npz_files = glob.glob(os.path.join(patent_embedding_folder, "*.npz"))

    IPC_3s_str = "G01N"
    IPC_3s = IPC_3s_str.split(',')
    # for file in npz_files:
    #     # 从文件名中提取IPC_3代码
    #     file_name = os.path.basename(file)
    #     if "patent_brief_" in file_name and "_embeddings_0.npz" in file_name:
    #         ipc_code = file_name.replace("patent_brief_", "").replace("_embeddings_0.npz", "")
    #         IPC_3s.append(ipc_code)

    # 进行保存的处理批次数量
    save_batch = 1000

    for ipc_code in tqdm(IPC_3s, desc="处理IPC_3s"):
        # 加载数据
        merged_df = load_data(os.path.join(patent_data_folder, 'patent_data_'+ipc_code+'_cleaned.csv'), 'ipc_categories_updated.csv')   ################
        # 创建索引映射表
        n = len(merged_df)
        # 创建特征矩阵
        feature_matrices_gpu, n = create_feature_matrices(merged_df, weights)
        # 加载专利的embedding npz
        brief_embeddings_array, brief_ids_array = load_patent_embeddings(os.path.join(patent_embedding_folder, 'patent_brief_'+ipc_code+'_embeddings_0.npz'))
        title_embeddings_array, title_ids_array = load_patent_embeddings(os.path.join(patent_embedding_folder, 'patent_title_'+ipc_code+'_embeddings_0.npz'))
        # 分批计算相似度矩阵
        print("开始计算分层相似度矩阵（GPU加速）...")
        batch_size = 300  # GPU内存允许的情况下可以增加批次大小  ################
        total_batches = (n + batch_size - 1) // batch_size

        similiarity_results = []
        # 串行处理各个批次
        for i in tqdm(range(total_batches), desc="处理批次"):
            result = process_batch_gpu(i, batch_size, n, feature_matrices_gpu, weights, brief_embeddings_array, title_embeddings_array, brief_ids_array.tolist())
            for res in result:
                similiarity_results.extend(res[:1001])
            # 判断是否保存
            if i > 0 and i % save_batch == 0:
                save_similiarity_results(similiarity_results, output_dir, ipc_code+'_'+str(i))
                similiarity_results = []
        # 保存剩余结果
        if len(similiarity_results) > 0:
            save_similiarity_results(similiarity_results, output_dir, ipc_code+'_'+str(total_batches))

if __name__ == "__main__":
    main()
