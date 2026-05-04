import argparse
import gc
import glob
import os

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import csr_matrix
from tqdm import tqdm

from config import (
    IPC_CATEGORIES_FILE,
    IPC_WEIGHTS,
    PATENT_DATA_DIR,
    PATENT_EMBEDDING_DIR,
    SIMILARITY_BATCH_SIZE,
    SIMILARITY_OUTPUT_DIR,
    SIMILARITY_SAVE_BATCH,
    SIMILARITY_THRESHOLD,
    SIMILARITY_WEIGHTS,
    TOP_K_NEIGHBORS,
)


def setup_gpu(gpu_id):
    cp.cuda.Device(gpu_id).use()
    device_name = cp.cuda.runtime.getDeviceProperties(gpu_id)['name'].decode()
    print(f"使用GPU {gpu_id}: {device_name}")


def load_data(patent_file, ipc_file):
    patent_df = pd.read_csv(patent_file)
    ipc_df = pd.read_csv(ipc_file)
    merged_df = pd.merge(patent_df, ipc_df, left_on='main_ipc', right_on='level_code', how='inner')
    needed_columns = ['id', 'title', 'brief', 'level1_code', 'level2_code',
                      'level3_code', 'level4_code', 'level5_code']
    return merged_df[needed_columns]


def create_feature_matrices(merged_df, weights):
    n = len(merged_df)
    feature_matrices_gpu = {}
    for level in weights:
        unique_values = merged_df[level].dropna().unique()
        code_to_idx = {code: idx for idx, code in enumerate(unique_values)}
        rows, cols, data = [], [], []
        for i, code in enumerate(merged_df[level]):
            if pd.notna(code):
                rows.append(i)
                cols.append(code_to_idx[code])
                data.append(1.0)
        cpu_matrix = csr_matrix(
            (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
            shape=(n, len(code_to_idx)),
        )
        feature_matrices_gpu[level] = cp_csr_matrix(cpu_matrix)
        del rows, cols, data
        gc.collect()
    return feature_matrices_gpu, n


def load_embeddings(file_name):
    data = cp.load(file_name)
    return data['embeddings'], data['ids']


def process_batch(i, batch_size, n, feature_matrices_gpu, weights,
                  brief_emb, title_emb, patent_ids,
                  threshold=None, top_k=None):
    start_i = i * batch_size
    end_i = min((i + 1) * batch_size, n)

    combined = None
    for level, weight in weights.items():
        batch_feat = feature_matrices_gpu[level][start_i:end_i]
        level_sim = batch_feat.dot(feature_matrices_gpu[level].T).multiply(weight)
        combined = level_sim if combined is None else combined + level_sim

    row_indices = cp.arange(end_i - start_i)
    diagonal_indices = cp.arange(start_i, end_i)
    combined[row_indices, diagonal_indices] = 1.0

    brief_sim = cp.dot(brief_emb[start_i:end_i], brief_emb.T)
    title_sim = cp.dot(title_emb[start_i:end_i], title_emb.T)

    score_matrix = (
        combined.toarray() * SIMILARITY_WEIGHTS['ipc']
        + brief_sim * SIMILARITY_WEIGHTS['brief']
        + title_sim * SIMILARITY_WEIGHTS['title']
    ).get()

    results = []
    _threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
    _top_k = TOP_K_NEIGHBORS if top_k is None else top_k
    for row_idx in range(score_matrix.shape[0]):
        row = score_matrix[row_idx]
        mask = row > _threshold
        valid = np.where(mask)[0]
        if len(valid) > _top_k:
            top_idx = valid[np.argsort(row[valid])[-_top_k:]]
            top_idx = top_idx[np.argsort(-row[top_idx])]
        else:
            top_idx = valid[np.argsort(-row[valid])]
        pid = patent_ids[int(row_idx + start_i)]
        results.append([
            (pid, patent_ids[int(col)], float(row[col])) for col in top_idx
        ])
    return results


def save_results(results, output_dir, tag):
    df = pd.DataFrame(results, columns=['patent_id', 'similar_patent_id', 'similarity_score'])
    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(f'{output_dir}/similiarity_results_{tag}.parquet', index=False)


def process_ipc(ipc_code, gpu_id):
    setup_gpu(gpu_id)

    merged_df = load_data(
        os.path.join(PATENT_DATA_DIR, f'patent_data_{ipc_code}_cleaned.csv'),
        IPC_CATEGORIES_FILE,
    )
    feature_matrices_gpu, n = create_feature_matrices(merged_df, IPC_WEIGHTS)

    brief_emb, brief_ids = load_embeddings(
        os.path.join(PATENT_EMBEDDING_DIR, f'patent_brief_{ipc_code}_embeddings_0.npz'))
    title_emb, _ = load_embeddings(
        os.path.join(PATENT_EMBEDDING_DIR, f'patent_title_{ipc_code}_embeddings_0.npz'))

    patent_ids = brief_ids.tolist()
    total_batches = (n + SIMILARITY_BATCH_SIZE - 1) // SIMILARITY_BATCH_SIZE
    accumulated = []

    for i in tqdm(range(total_batches), desc=f"{ipc_code} 批次"):
        batch_results = process_batch(
            i, SIMILARITY_BATCH_SIZE, n, feature_matrices_gpu, IPC_WEIGHTS,
            brief_emb, title_emb, patent_ids,
        )
        for res in batch_results:
            accumulated.extend(res[:TOP_K_NEIGHBORS + 1])
        if i > 0 and i % SIMILARITY_SAVE_BATCH == 0:
            save_results(accumulated, SIMILARITY_OUTPUT_DIR, f'{ipc_code}_{i}')
            accumulated = []

    if accumulated:
        save_results(accumulated, SIMILARITY_OUTPUT_DIR, f'{ipc_code}_{total_batches}')


def worker(gpu_id, ipc_queue):
    """每个 GPU 对应一个 worker，持续从队列取 IPC 直到队列为空。"""
    while True:
        try:
            ipc_code = ipc_queue.get_nowait()
        except Exception:
            break
        print(f"[GPU {gpu_id}] 开始处理 {ipc_code}")
        process_ipc(ipc_code, gpu_id)
        print(f"[GPU {gpu_id}] 完成 {ipc_code}")


def scan_ipc_codes():
    """从 patent_embedding/ 自动扫描已有向量的 IPC 代码。"""
    import glob
    codes = set()
    for f in glob.glob(os.path.join(PATENT_EMBEDDING_DIR, 'patent_brief_*_embeddings_0.npz')):
        name = os.path.basename(f)
        code = name.replace('patent_brief_', '').replace('_embeddings_0.npz', '')
        codes.add(code)
    return sorted(codes)


if __name__ == '__main__':
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='专利相似度计算')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU 编号，逗号分隔，如 0,1,2')
    parser.add_argument('--ipc', type=str, default=None,
                        help='IPC 代码，逗号分隔；不指定则自动扫描 patent_embedding/')
    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    ipc_codes = [c.strip() for c in args.ipc.split(',')] if args.ipc else scan_ipc_codes()

    if not ipc_codes:
        print("未找到任何 IPC 代码，请确认 patent_embedding/ 目录或通过 --ipc 指定。")
        exit(1)

    print(f"共 {len(ipc_codes)} 个 IPC，使用 GPU: {gpu_ids}")

    queue = mp.Queue()
    for code in ipc_codes:
        queue.put(code)

    processes = [mp.Process(target=worker, args=(gpu_id, queue)) for gpu_id in gpu_ids]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("所有 IPC 处理完成。")
