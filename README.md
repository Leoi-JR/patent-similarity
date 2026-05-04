# 专利相似度计算系统

基于 IPC 层级结构 + 语义向量的专利相似度批量计算流水线。

## 原理

相似度分数由三部分加权合成：

```
score = IPC结构相似度 × 0.4 + 摘要向量余弦相似度 × 0.4 + 标题向量余弦相似度 × 0.2
```

IPC 结构相似度为 level3/4/5 稀疏独热重叠的加权求和（权重 0.2 / 0.3 / 0.5）。只保留 score > 0.75 的专利对，每个专利最多保留 1000 个最近邻。

## 环境准备

```bash
conda create -n patent python=3.10
/opt/conda/envs/patent/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
/opt/conda/envs/patent/bin/pip install modelscope transformers==4.44.2 pandas numpy flask tqdm requests scipy pyarrow cupy-cuda12x pytest
```

## 数据目录

```
patent_data/             # 输入：patent_data_<IPC>_cleaned.csv（列：id, title, brief, main_ipc）
                         #       ipc_categories_updated.csv（IPC 层级查找表）
patent_embedding/        # 中间结果：patent_title_<IPC>_embeddings_0.npz / patent_brief_<IPC>_embeddings_0.npz
similarity_results_gpu/  # 输出：similiarity_results_<IPC>_<batch>.parquet
model/                   # ModelScope 模型缓存
```

## 运行流水线

### 第一步：启动向量服务

每个 GPU 启动一个实例，端口与 `config.py` 中 `EMBEDDING_SERVERS` 保持一致：

```bash
/opt/conda/envs/patent/bin/python embedding_server.py --port 5000 --gpu 0
/opt/conda/envs/patent/bin/python embedding_server.py --port 5001 --gpu 1
```

### 第二步：批量生成向量

等服务启动后运行（自动扫描 `patent_data/` 下所有 IPC，多服务并行）：

```bash
/opt/conda/envs/patent/bin/python generate_embedding.py
```

### 第三步：计算相似度

支持多 GPU 自动调度，`--ipc` 不指定时自动扫描 `patent_embedding/`：

```bash
# 指定 IPC
/opt/conda/envs/patent/bin/python compute_similarity.py --gpus 0,1,2 --ipc G06F,G01N,H04R

# 自动扫描所有已有向量的 IPC
/opt/conda/envs/patent/bin/python compute_similarity.py --gpus 0,1,2
```

## 配置

所有参数集中在 `config.py`，常用配置项：

| 参数 | 说明 |
|---|---|
| `EMBEDDING_SERVERS` | 向量服务地址列表，与启动的服务端口对应 |
| `SIMILARITY_THRESHOLD` | 相似度过滤阈值（默认 0.75） |
| `SIMILARITY_BATCH_SIZE` | 相似度计算批次大小，受 GPU 显存限制（默认 300） |
| `IPC_WEIGHTS` | IPC 各层级权重 |
| `SIMILARITY_WEIGHTS` | IPC 结构 / 摘要 / 标题三分量权重 |

## 测试

```bash
# 冒烟测试（CPU，无需 GPU，随时可跑）
/opt/conda/envs/patent/bin/pytest tests/test_compute_similarity.py -v

# 端到端集成测试（需要 GPU 0 和 GPU 1）
/opt/conda/envs/patent/bin/pytest tests/test_e2e.py -v
```

测试输出保存在 `tests/output/`。

## 文件结构

```
├── config.py              # 所有配置
├── embedding_server.py    # 向量服务（--port / --gpu 参数化）
├── generate_embedding.py  # 批量向量化客户端
├── compute_similarity.py  # 相似度计算（--gpus / --ipc 参数化）
├── tests/
│   ├── conftest.py
│   ├── fixtures/          # 测试用小批量数据
│   ├── test_compute_similarity.py  # 冒烟测试
│   └── test_e2e.py        # 端到端集成测试
└── scripts/
    ├── generate_sim0.py   # 单服务调试用
    ├── transfer.py        # CSV 转 NPZ 工具
    └── notebooks/         # 探索性分析
```
