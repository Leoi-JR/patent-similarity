"""
冒烟测试：在 CPU 环境下用小批量 fixture 验证真实相似度计算主流程可用。

运行方式：
    pytest tests/test_compute_similarity.py -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import compute_similarity

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
PATENT_FILE = os.path.join(FIXTURES_DIR, 'patents_100.csv')
IPC_FILE = os.path.join(FIXTURES_DIR, 'ipc_categories.csv')
BRIEF_EMB_FILE = os.path.join(FIXTURES_DIR, 'patent_brief_TEST_embeddings_0.npz')
TITLE_EMB_FILE = os.path.join(FIXTURES_DIR, 'patent_title_TEST_embeddings_0.npz')

pytestmark = [pytest.mark.smoke]


@pytest.fixture
def cpu_cupy(monkeypatch):
    class FakeArray(np.ndarray):
        def get(self):
            return np.asarray(self)

    def as_fake(array_like, dtype=None):
        array = np.asarray(array_like, dtype=dtype)
        return array.view(FakeArray)

    class FakeDevice:
        def __init__(self, gpu_id):
            self.gpu_id = gpu_id

        def use(self):
            return None

    class FakeRuntime:
        @staticmethod
        def getDeviceProperties(gpu_id):
            return {'name': f'CPU-{gpu_id}'.encode()}

    class FakeCuda:
        Device = FakeDevice
        runtime = FakeRuntime()

    class FakeSparseMatrix:
        def __init__(self, matrix):
            self.matrix = matrix

        def __getitem__(self, key):
            return FakeSparseMatrix(self.matrix[key])

        @property
        def T(self):
            return FakeSparseMatrix(self.matrix.T)

        def dot(self, other):
            other_matrix = other.matrix if isinstance(other, FakeSparseMatrix) else other
            return FakeSparseMatrix(self.matrix.dot(other_matrix))

        def multiply(self, value):
            return FakeSparseMatrix(self.matrix.multiply(value))

        def toarray(self):
            return as_fake(self.matrix.toarray(), dtype=np.float32)

        def __add__(self, other):
            return FakeSparseMatrix(self.matrix + other.matrix)

        def __setitem__(self, key, value):
            self.matrix[key] = value

    class FakeCupy:
        float32 = np.float32
        cuda = FakeCuda()

        @staticmethod
        def load(file_name):
            return np.load(file_name)

        @staticmethod
        def array(array_like, dtype=None):
            return as_fake(array_like, dtype=dtype)

        @staticmethod
        def arange(*args, **kwargs):
            return as_fake(np.arange(*args, **kwargs))

        @staticmethod
        def dot(a, b):
            return as_fake(np.dot(np.asarray(a), np.asarray(b)), dtype=np.float32)

        @staticmethod
        def clip(a, a_min, a_max):
            return as_fake(np.clip(np.asarray(a), a_min, a_max), dtype=np.float32)

    monkeypatch.setattr(compute_similarity, 'cp', FakeCupy)
    monkeypatch.setattr(compute_similarity, 'cp_csr_matrix', lambda matrix: FakeSparseMatrix(matrix))


def test_load_data_uses_real_module_path():
    df = compute_similarity.load_data(PATENT_FILE, IPC_FILE)
    assert len(df) > 0, 'merged_df 为空'
    for col in ['id', 'level3_code', 'level4_code', 'level5_code']:
        assert col in df.columns, f'缺少列: {col}'


def test_create_feature_matrices_uses_real_module_path(cpu_cupy):
    df = compute_similarity.load_data(PATENT_FILE, IPC_FILE)
    matrices, n = compute_similarity.create_feature_matrices(df, compute_similarity.IPC_WEIGHTS)
    assert n == len(df)
    for level in compute_similarity.IPC_WEIGHTS:
        assert level in matrices


def test_process_ipc_writes_expected_parquet(cpu_cupy):
    with tempfile.TemporaryDirectory() as tmpdir:
        patent_dir = os.path.join(tmpdir, 'patent_data')
        embedding_dir = os.path.join(tmpdir, 'patent_embedding')
        output_dir = os.path.join(tmpdir, 'similarity_results')
        os.makedirs(patent_dir, exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        pd.read_csv(PATENT_FILE).to_csv(
            os.path.join(patent_dir, 'patent_data_TEST_cleaned.csv'),
            index=False,
        )
        pd.read_csv(IPC_FILE).to_csv(
            os.path.join(patent_dir, 'ipc_categories.csv'),
            index=False,
        )

        for src, name in [
            (BRIEF_EMB_FILE, 'patent_brief_TEST_embeddings_0.npz'),
            (TITLE_EMB_FILE, 'patent_title_TEST_embeddings_0.npz'),
        ]:
            data = np.load(src)
            np.savez_compressed(
                os.path.join(embedding_dir, name),
                embeddings=data['embeddings'].astype(np.float32),
                ids=data['ids'],
            )

        compute_similarity.process_ipc(
            'TEST',
            gpu_id=0,
            patent_data_dir=patent_dir,
            patent_embedding_dir=embedding_dir,
            ipc_categories_file=os.path.join(patent_dir, 'ipc_categories.csv'),
            output_dir=output_dir,
            threshold=0.0,
            batch_size=20,
            top_k=50,
        )

        parquet_files = sorted(
            f for f in os.listdir(output_dir)
            if f.endswith('.parquet')
        )
        assert parquet_files, '没有生成 parquet 文件'

        result_df = pd.read_parquet(os.path.join(output_dir, parquet_files[0]))
        assert list(result_df.columns) == ['patent_id', 'similar_patent_id', 'similarity_score']
        assert len(result_df) > 0
        assert np.isfinite(result_df['similarity_score']).all()
        assert result_df['similarity_score'].between(0.0, 1.0).all()
        assert (result_df['patent_id'] != result_df['similar_patent_id']).all()


def test_process_ipc_rejects_misaligned_embedding_ids(cpu_cupy):
    with tempfile.TemporaryDirectory() as tmpdir:
        patent_dir = os.path.join(tmpdir, 'patent_data')
        embedding_dir = os.path.join(tmpdir, 'patent_embedding')
        output_dir = os.path.join(tmpdir, 'similarity_results')
        os.makedirs(patent_dir, exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        pd.read_csv(PATENT_FILE).to_csv(
            os.path.join(patent_dir, 'patent_data_TEST_cleaned.csv'),
            index=False,
        )
        pd.read_csv(IPC_FILE).to_csv(
            os.path.join(patent_dir, 'ipc_categories.csv'),
            index=False,
        )

        brief = np.load(BRIEF_EMB_FILE)
        title = np.load(TITLE_EMB_FILE)
        np.savez_compressed(
            os.path.join(embedding_dir, 'patent_brief_TEST_embeddings_0.npz'),
            embeddings=brief['embeddings'].astype(np.float32),
            ids=brief['ids'],
        )
        np.savez_compressed(
            os.path.join(embedding_dir, 'patent_title_TEST_embeddings_0.npz'),
            embeddings=title['embeddings'].astype(np.float32),
            ids=title['ids'][::-1],
        )

        with pytest.raises(ValueError, match='embedding id 顺序不一致'):
            compute_similarity.process_ipc(
                'TEST',
                gpu_id=0,
                patent_data_dir=patent_dir,
                patent_embedding_dir=embedding_dir,
                ipc_categories_file=os.path.join(patent_dir, 'ipc_categories.csv'),
                output_dir=output_dir,
                batch_size=20,
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
