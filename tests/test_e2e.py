"""
集成测试：启动真实 embedding 服务，跑通向量生成和相似度计算链路并验证输出。

依赖：
    - 当前环境安装 torch / modelscope / cupy
    - 可启动 embedding_server.py
    - 至少 1 个可用计算设备；若无 CUDA，则服务会退回 CPU

运行方式：
    pytest tests/test_e2e.py -v
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest
import requests
import torch

import config
import generate_embedding
from compute_similarity import process_ipc
from generate_embedding import process_file

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
PATENT_FILE = os.path.join(FIXTURES_DIR, 'patents_100.csv')
IPC_FILE = os.path.join(FIXTURES_DIR, 'ipc_categories.csv')
PYTHON = sys.executable

pytestmark = [pytest.mark.integration]


def _pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _wait_for_server(url, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.post(f'{url}/embed', json={'texts': ['ping']}, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


@pytest.fixture(scope='session')
def runtime_device_id():
    return 0 if torch.cuda.is_available() else None


@pytest.fixture(scope='session')
def embedding_server(runtime_device_id):
    server_script = os.path.join(os.path.dirname(__file__), '..', 'embedding_server.py')
    port = _pick_free_port()
    base_url = f'http://127.0.0.1:{port}'
    env = os.environ.copy()
    env['MODELSCOPE_CACHE'] = config.MODEL_CACHE_DIR

    command = [PYTHON, server_script, '--port', str(port)]
    if runtime_device_id is not None:
        command.extend(['--gpu', str(runtime_device_id)])

    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        assert _wait_for_server(base_url), f'embedding 服务启动超时: {base_url}'
        yield {'base_url': base_url, 'gpu_id': runtime_device_id}
    finally:
        process.terminate()
        process.wait(timeout=30)


@pytest.fixture
def test_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        patent_dir = os.path.join(tmpdir, 'patent_data')
        embedding_dir = os.path.join(tmpdir, 'patent_embedding')
        similarity_dir = os.path.join(tmpdir, 'similarity_results')
        os.makedirs(patent_dir, exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)
        os.makedirs(similarity_dir, exist_ok=True)

        shutil.copy(PATENT_FILE, os.path.join(patent_dir, 'patent_data_TEST_cleaned.csv'))
        shutil.copy(IPC_FILE, os.path.join(patent_dir, 'ipc_categories.csv'))

        yield {
            'patent_dir': patent_dir,
            'embedding_dir': embedding_dir,
            'similarity_dir': similarity_dir,
            'ipc_file': os.path.join(patent_dir, 'ipc_categories.csv'),
        }


def test_embedding_server_response(embedding_server):
    response = requests.post(
        f"{embedding_server['base_url']}/embed",
        json={'texts': ['专利测试文本']},
        timeout=30,
    )
    assert response.status_code == 200
    embeddings = response.json()['embeddings']
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0


def test_end_to_end_pipeline(embedding_server, test_workspace):
    process_file(
        'TEST',
        base_url=embedding_server['base_url'],
        patent_data_dir=test_workspace['patent_dir'],
        patent_embedding_dir=test_workspace['embedding_dir'],
        batch_sizes={'title': 50, 'brief': 50},
        save_batch_size=10,
    )

    outputs = {}
    for field in ['title', 'brief']:
        npz_path = os.path.join(test_workspace['embedding_dir'], f'patent_{field}_TEST_embeddings_0.npz')
        assert os.path.exists(npz_path), f'缺少输出文件: {npz_path}'
        data = np.load(npz_path)
        outputs[field] = data
        assert 'embeddings' in data and 'ids' in data
        assert data['embeddings'].ndim == 2
        assert data['embeddings'].shape[0] == data['ids'].shape[0]
        assert len(data['ids']) > 0

    assert outputs['title']['ids'].tolist() == outputs['brief']['ids'].tolist()

    if embedding_server['gpu_id'] is None:
        pytest.skip('当前环境无 CUDA，跳过 GPU 相似度计算集成步骤')

    process_ipc(
        'TEST',
        gpu_id=embedding_server['gpu_id'],
        patent_data_dir=test_workspace['patent_dir'],
        patent_embedding_dir=test_workspace['embedding_dir'],
        ipc_categories_file=test_workspace['ipc_file'],
        output_dir=test_workspace['similarity_dir'],
        threshold=0.75,
        batch_size=20,
        top_k=50,
    )

    parquet_files = sorted(
        f for f in os.listdir(test_workspace['similarity_dir'])
        if f.endswith('.parquet')
    )
    assert parquet_files, '没有生成 parquet 文件'

    result_df = pd.read_parquet(os.path.join(test_workspace['similarity_dir'], parquet_files[0]))
    assert list(result_df.columns) == ['patent_id', 'similar_patent_id', 'similarity_score']
    assert len(result_df) > 0
    assert np.isfinite(result_df['similarity_score']).all()
    assert result_df['similarity_score'].between(0.0, 1.0).all()
    assert (result_df['patent_id'] != result_df['similar_patent_id']).all()


def test_embedding_generation_fails_fast_on_batch_error(monkeypatch, test_workspace):
    def fail_embed(text_list, base_url):
        raise requests.RequestException('boom')

    monkeypatch.setattr(generate_embedding, 'embed', fail_embed)

    with pytest.raises(requests.RequestException, match='boom'):
        process_file(
            'TEST',
            base_url='http://127.0.0.1:9',
            patent_data_dir=test_workspace['patent_dir'],
            patent_embedding_dir=test_workspace['embedding_dir'],
            batch_sizes={'title': 50},
            save_batch_size=10,
        )

    assert not os.listdir(test_workspace['embedding_dir'])


def test_generate_embedding_main_raises_on_failed_ipc(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_embedding, 'PATENT_DATA_DIR', str(tmp_path / 'patent_data'))
    monkeypatch.setattr(generate_embedding, 'PATENT_EMBEDDING_DIR', str(tmp_path / 'patent_embedding'))
    monkeypatch.setattr(generate_embedding, 'EMBEDDING_SERVERS', ['http://127.0.0.1:9'])

    patent_dir = tmp_path / 'patent_data'
    patent_dir.mkdir()
    pd.DataFrame([
        {'id': '1', 'title': 'a', 'brief': 'b', 'main_ipc': 'TEST'}
    ]).to_csv(patent_dir / 'patent_data_TEST_cleaned.csv', index=False)

    def fail_process_file(ipc_code, base_url):
        raise RuntimeError(f'{ipc_code} failed')

    monkeypatch.setattr(generate_embedding, 'process_file', fail_process_file)

    with pytest.raises(RuntimeError, match='以下 IPC 处理失败: TEST'):
        generate_embedding.main()
