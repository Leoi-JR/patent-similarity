"""
端到端集成测试：启动真实 embedding 服务，跑完整流水线并验证输出。

依赖：
    - GPU 可用（至少 2 块，GPU 0 和 GPU 1）
    - /opt/conda/envs/patent/bin/python 已安装 torch/modelscope/cupy

运行方式：
    /opt/conda/envs/patent/bin/pytest tests/test_e2e.py -v
"""

import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
import requests

import config
from compute_similarity import process_ipc
from generate_embedding import process_file

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
PATENT_FILE = os.path.join(FIXTURES_DIR, 'patents_100.csv')
IPC_FILE = os.path.join(FIXTURES_DIR, 'ipc_categories.csv')
PYTHON = sys.executable

TEST_SERVERS = [
    {"port": 5000, "gpu": 0},
    {"port": 5001, "gpu": 1},
]
TEST_SERVER_URLS = [f"http://localhost:{s['port']}" for s in TEST_SERVERS]


# ---------- fixtures ----------

def _wait_for_server(url, timeout=120):
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


@pytest.fixture(scope="session")
def embedding_servers():
    """启动 embedding 服务，整个测试 session 共享，结束后关闭。"""
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

    for url in TEST_SERVER_URLS:
        assert _wait_for_server(url, timeout=120), f"embedding 服务启动超时: {url}"

    yield TEST_SERVER_URLS

    for p in procs:
        p.terminate()
        p.wait()


@pytest.fixture(scope="session")
def output_dirs():
    """准备固定输出目录，返回各子目录路径。"""
    patent_dir = os.path.join(OUTPUT_DIR, 'patent_data')
    emb_dir = os.path.join(OUTPUT_DIR, 'patent_embedding')
    sim_dir = os.path.join(OUTPUT_DIR, 'similarity_results')
    for d in [patent_dir, emb_dir, sim_dir]:
        os.makedirs(d, exist_ok=True)
    return {"patent": patent_dir, "embedding": emb_dir, "similarity": sim_dir}


@pytest.fixture(scope="session")
def embeddings(embedding_servers, output_dirs):
    """生成 title/brief 向量，供后续测试使用。"""
    patent_dir = output_dirs["patent"]
    emb_dir = output_dirs["embedding"]

    df = pd.read_csv(PATENT_FILE).fillna("")
    df.to_csv(os.path.join(patent_dir, 'patent_data_TEST_cleaned.csv'), index=False)

    process_file(
        'TEST', base_url=embedding_servers[0],
        patent_data_dir=patent_dir,
        patent_embedding_dir=emb_dir,
        batch_sizes={"title": 50, "brief": 50},
        save_batch_size=10,
    )
    return emb_dir


# ---------- 测试用例 ----------

def test_embedding_server_response(embedding_servers):
    """每个服务都能正常返回正确维度的向量。"""
    for url in embedding_servers:
        r = requests.post(f"{url}/embed", json={"texts": ["专利测试文本"]})
        assert r.status_code == 200
        emb = r.json()["embeddings"]
        assert len(emb) == 1
        assert len(emb[0]) > 0


def test_embeddings_output_files(embeddings):
    """title/brief npz 文件存在，且结构正确。"""
    for field in ['title', 'brief']:
        npz_path = os.path.join(embeddings, f'patent_{field}_TEST_embeddings_0.npz')
        assert os.path.exists(npz_path), f"缺少输出文件: {npz_path}"
        data = np.load(npz_path)
        assert 'embeddings' in data and 'ids' in data
        assert data['embeddings'].ndim == 2
        assert data['embeddings'].shape[0] == data['ids'].shape[0]
        assert len(data['ids']) > 0


def test_similarity_output(embeddings, output_dirs):
    """相似度计算生成 parquet，格式和内容正确。"""
    patent_dir = output_dirs["patent"]
    sim_dir = output_dirs["similarity"]
    shutil.copy(IPC_FILE, os.path.join(patent_dir, 'ipc_categories.csv'))

    process_ipc(
        'TEST', gpu_id=0,
        patent_data_dir=patent_dir,
        patent_embedding_dir=embeddings,
        ipc_categories_file=os.path.join(patent_dir, 'ipc_categories.csv'),
        output_dir=sim_dir,
        threshold=0,
        batch_size=20,
    )

    parquet_files = [f for f in os.listdir(sim_dir) if f.endswith('.parquet')]
    assert len(parquet_files) > 0, "没有生成 parquet 文件"

    df = pd.read_parquet(os.path.join(sim_dir, parquet_files[0]))
    assert list(df.columns) == ['patent_id', 'similar_patent_id', 'similarity_score']
    assert len(df) > 0
    assert (df['patent_id'] != df['similar_patent_id']).any(), "结果中应包含不同专利之间的相似度"
