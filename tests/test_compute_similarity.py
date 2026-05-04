"""
冒烟测试：用小批量 fixture 数据验证相似度计算流程的完整可用性（CPU，无需 GPU）。

运行方式：
    /opt/conda/envs/patent/bin/pytest tests/test_compute_similarity.py -v
"""

import gc
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import config

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
PATENT_FILE = os.path.join(FIXTURES_DIR, 'patents_100.csv')
IPC_FILE = os.path.join(FIXTURES_DIR, 'ipc_categories.csv')
BRIEF_EMB_FILE = os.path.join(FIXTURES_DIR, 'patent_brief_TEST_embeddings_0.npz')
TITLE_EMB_FILE = os.path.join(FIXTURES_DIR, 'patent_title_TEST_embeddings_0.npz')


# ---------- CPU 版核心函数（与 compute_similarity.py 逻辑一致，用 numpy 替代 cupy） ----------

def load_data(patent_file, ipc_file):
    patent_df = pd.read_csv(patent_file)
    ipc_df = pd.read_csv(ipc_file)
    merged_df = pd.merge(patent_df, ipc_df, left_on='main_ipc', right_on='level_code', how='inner')
    needed = ['id', 'title', 'brief', 'level1_code', 'level2_code',
              'level3_code', 'level4_code', 'level5_code']
    return merged_df[needed]


def create_feature_matrices_cpu(merged_df, weights):
    n = len(merged_df)
    matrices = {}
    for level in weights:
        unique_values = merged_df[level].dropna().unique()
        code_to_idx = {code: idx for idx, code in enumerate(unique_values)}
        rows, cols, data = [], [], []
        for i, code in enumerate(merged_df[level]):
            if pd.notna(code):
                rows.append(i)
                cols.append(code_to_idx[code])
                data.append(1.0)
        matrices[level] = csr_matrix(
            (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
            shape=(n, len(code_to_idx)),
        )
        del rows, cols, data
        gc.collect()
    return matrices, n


def process_batch_cpu(start_i, end_i, n, matrices, weights,
                      brief_emb, title_emb, patent_ids):
    combined = None
    for level, weight in weights.items():
        batch = matrices[level][start_i:end_i]
        sim = batch.dot(matrices[level].T).toarray() * weight
        combined = sim if combined is None else combined + sim

    row_idx = np.arange(end_i - start_i)
    combined[row_idx, np.arange(start_i, end_i)] = 1.0

    brief_sim = brief_emb[start_i:end_i] @ brief_emb.T
    title_sim = title_emb[start_i:end_i] @ title_emb.T

    scores = (combined * config.SIMILARITY_WEIGHTS['ipc']
              + brief_sim * config.SIMILARITY_WEIGHTS['brief']
              + title_sim * config.SIMILARITY_WEIGHTS['title'])

    results = []
    for r in range(scores.shape[0]):
        row = scores[r]
        valid = np.where(row > config.SIMILARITY_THRESHOLD)[0]
        if len(valid) > config.TOP_K_NEIGHBORS:
            top = valid[np.argsort(row[valid])[-config.TOP_K_NEIGHBORS:]]
            top = top[np.argsort(-row[top])]
        else:
            top = valid[np.argsort(-row[valid])]
        pid = patent_ids[start_i + r]
        results.extend((pid, patent_ids[int(c)], float(row[c])) for c in top)
    return results


# ---------- 测试用例 ----------

def test_load_data():
    df = load_data(PATENT_FILE, IPC_FILE)
    assert len(df) > 0, "merged_df 为空"
    for col in ['id', 'level3_code', 'level4_code', 'level5_code']:
        assert col in df.columns, f"缺少列: {col}"


def test_feature_matrices():
    df = load_data(PATENT_FILE, IPC_FILE)
    matrices, n = create_feature_matrices_cpu(df, config.IPC_WEIGHTS)
    assert n == len(df)
    for level in config.IPC_WEIGHTS:
        assert level in matrices
        assert matrices[level].shape[0] == n


def test_embeddings_shape():
    brief = np.load(BRIEF_EMB_FILE)
    title = np.load(TITLE_EMB_FILE)
    assert brief['embeddings'].shape[1] == title['embeddings'].shape[1], "brief/title 向量维度不一致"
    assert len(brief['ids']) == len(title['ids']), "brief/title id 数量不一致"


def test_similarity_output():
    df = load_data(PATENT_FILE, IPC_FILE)
    matrices, n = create_feature_matrices_cpu(df, config.IPC_WEIGHTS)

    brief_data = np.load(BRIEF_EMB_FILE)
    title_data = np.load(TITLE_EMB_FILE)
    brief_emb = brief_data['embeddings'].astype(np.float32)
    title_emb = title_data['embeddings'].astype(np.float32)
    patent_ids = brief_data['ids'].tolist()

    # 只跑前两批
    batch_size = 10
    results = []
    for i in range(2):
        start_i = i * batch_size
        end_i = min(start_i + batch_size, n)
        results.extend(process_batch_cpu(start_i, end_i, n, matrices, config.IPC_WEIGHTS,
                                         brief_emb, title_emb, patent_ids))

    assert len(results) > 0, "没有输出任何相似度结果"
    pid, spid, score = results[0]
    assert 0.0 <= score <= 1.0, f"相似度分数超出范围: {score}"

    # 验证保存为 parquet 格式正常
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, 'test_output.parquet')
        result_df = pd.DataFrame(results, columns=['patent_id', 'similar_patent_id', 'similarity_score'])
        result_df.to_parquet(out_file, index=False)
        loaded = pd.read_parquet(out_file)
        assert list(loaded.columns) == ['patent_id', 'similar_patent_id', 'similarity_score']
        assert len(loaded) == len(results)



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
