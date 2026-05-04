"""
端到端集成测试：启动真实 embedding 服务，跑完整流水线并验证输出。

依赖：
    - /opt/conda/envs/patent/bin/python（已安装 torch/modelscope/cupy）
    - GPU 可用（至少 2 块，GPU 0 和 GPU 1）

运行方式：
    /opt/conda/envs/patent/bin/python tests/test_e2e.py
"""

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
PATENT_FILE = os.path.join(FIXTURES_DIR, 'patents_100.csv')
IPC_FILE = os.path.join(FIXTURES_DIR, 'ipc_categories.csv')
PYTHON = sys.executable

# 测试用的两个 embedding 服务
TEST_SERVERS = [
    {"port": 5000, "gpu": 0},
    {"port": 5001, "gpu": 1},
]
TEST_SERVER_URLS = [f"http://localhost:{s['port']}" for s in TEST_SERVERS]


# ---------- 工具函数 ----------

def wait_for_server(url, timeout=120):
    """等待服务启动就绪。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.post(f"{url}/embed", json={"texts": ["ping"]}, timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def start_servers():
    """启动 embedding 服务子进程，返回进程列表。"""
    server_script = os.path.join(os.path.dirname(__file__), '..', 'embedding_server.py')
    procs = []
    for s in TEST_SERVERS:
        env = os.environ.copy()
        env['MODELSCOPE_CACHE'] = config.MODEL_CACHE_DIR
        p = subprocess.Popen(
            [PYTHON, server_script, '--port', str(s['port']), '--gpu', str(s['gpu'])],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)
    return procs


def stop_servers(procs):
    for p in procs:
        p.terminate()
        p.wait()


# ---------- 测试用例 ----------

def test_embedding_servers(procs):
    """验证两个服务都能正常响应。"""
    for url in TEST_SERVER_URLS:
        assert wait_for_server(url, timeout=120), f"服务未就绪: {url}"
        r = requests.post(f"{url}/embed", json={"texts": ["专利测试文本"]})
        assert r.status_code == 200
        emb = r.json()["embeddings"]
        assert len(emb) == 1
        assert len(emb[0]) == 1024
    print(f"  [PASS] embedding 服务正常，向量维度: 1024")


def test_generate_embeddings():
    """用 fixture 数据直接调用 generate_embedding 函数生成 npz，验证输出。"""
    from generate_embedding import process_embeddings

    patent_dir = os.path.join(OUTPUT_DIR, 'patent_data')
    emb_dir = os.path.join(OUTPUT_DIR, 'patent_embedding')
    os.makedirs(patent_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    df = pd.read_csv(PATENT_FILE).fillna("")
    df.to_csv(os.path.join(patent_dir, 'patent_data_TEST_cleaned.csv'), index=False)

    for field, batch_size in [('title', 50), ('brief', 50)]:
        output_file = os.path.join(emb_dir, f'patent_{field}_TEST_embeddings.npz')
        process_embeddings(
            df, field, 'id', batch_size, output_file,
            base_url=TEST_SERVER_URLS[0],
            save_batch_size=10,
        )

    for field in ['title', 'brief']:
        npz_path = os.path.join(emb_dir, f'patent_{field}_TEST_embeddings_0.npz')
        assert os.path.exists(npz_path), f"缺少输出文件: {npz_path}"
        data = np.load(npz_path)
        assert 'embeddings' in data and 'ids' in data
        assert data['embeddings'].shape[1] == 1024
        assert len(data['ids']) > 0

    print(f"  [PASS] generate_embedding: title/brief npz 生成正常，维度=1024")
    return emb_dir


def test_compute_similarity(emb_dir):
    """用生成的 npz 跑 compute_similarity，验证 parquet 输出。"""
    import shutil
    from compute_similarity import load_data, create_feature_matrices, load_embeddings, process_batch, save_results
    import cupy as cp

    patent_dir = os.path.join(OUTPUT_DIR, 'patent_data')
    out_dir = os.path.join(OUTPUT_DIR, 'similarity_results')
    os.makedirs(out_dir, exist_ok=True)

    shutil.copy(IPC_FILE, os.path.join(patent_dir, 'ipc_categories.csv'))

    # 集成测试不做业务过滤，只验证流程和格式
    config.SIMILARITY_THRESHOLD = 0

    cp.cuda.Device(0).use()
    merged_df = load_data(
        os.path.join(patent_dir, 'patent_data_TEST_cleaned.csv'),
        os.path.join(patent_dir, 'ipc_categories.csv'),
    )
    assert len(merged_df) > 0, "merged_df 为空，main_ipc 与 ipc_categories 无交集"

    feature_matrices_gpu, n = create_feature_matrices(merged_df, config.IPC_WEIGHTS)
    brief_emb, brief_ids = load_embeddings(os.path.join(emb_dir, 'patent_brief_TEST_embeddings_0.npz'))
    title_emb, _ = load_embeddings(os.path.join(emb_dir, 'patent_title_TEST_embeddings_0.npz'))

    patent_ids = brief_ids.tolist()
    batch_size = 20
    total_batches = (n + batch_size - 1) // batch_size
    accumulated = []
    for i in range(total_batches):
        for res in process_batch(i, batch_size, n, feature_matrices_gpu, config.IPC_WEIGHTS,
                                 brief_emb, title_emb, patent_ids):
            accumulated.extend(res[:config.TOP_K_NEIGHBORS + 1])

    assert len(accumulated) > 0, "没有输出任何相似度结果"
    save_results(accumulated, out_dir, 'TEST_final')

    parquet_files = [f for f in os.listdir(out_dir) if f.endswith('.parquet')]
    assert len(parquet_files) > 0
    df = pd.read_parquet(os.path.join(out_dir, parquet_files[0]))
    assert list(df.columns) == ['patent_id', 'similar_patent_id', 'similarity_score']
    assert len(df) > 1, "过滤已关闭，结果应包含多个专利对"

    print(f"  [PASS] compute_similarity: {len(df)} 条结果，已保存至 {out_dir}")


# ---------- 主入口 ----------

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== 端到端集成测试 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    print("启动 embedding 服务（GPU 0, 1）...")
    procs = start_servers()

    passed = 0
    total = 3
    emb_dir = None

    try:
        try:
            test_embedding_servers(procs)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_embedding_servers: {e}")

        try:
            emb_dir = test_generate_embeddings()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_generate_embeddings: {e}")

        if emb_dir:
            try:
                test_compute_similarity(emb_dir)
                passed += 1
            except Exception as e:
                import traceback
                print(f"  [FAIL] test_compute_similarity: {e}")
                traceback.print_exc()
        else:
            print("  [SKIP] test_compute_similarity: 上一步未生成 embedding")
            total -= 1
    finally:
        print("关闭 embedding 服务...")
        stop_servers(procs)

    print(f"=== 完成：{passed}/{total} 通过 ===")
