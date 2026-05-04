# 向量服务地址（与 embedding_server.py 启动时的端口对应）
EMBEDDING_SERVERS = [
    "http://localhost:5000",
    "http://localhost:5001",
    "http://localhost:5002",
    "http://localhost:5003",
    "http://localhost:5005",
]

# 各字段的向量化批次大小（受模型 sequence_length 和显存限制）
EMBEDDING_BATCH_SIZES = {
    "title": 5000,
    "brief": 300,
}

# 向量化时每攒多少批次写一次 npz 文件
EMBEDDING_SAVE_BATCH_SIZE = 1000

# 数据目录
PATENT_DATA_DIR = "patent_data"
PATENT_EMBEDDING_DIR = "patent_embedding"
SIMILARITY_OUTPUT_DIR = "similarity_results_gpu"
IPC_CATEGORIES_FILE = "patent_data/ipc_categories_updated.csv"
MODEL_CACHE_DIR = "/workspace/patent/model"

# 相似度计算：IPC 层级权重
IPC_WEIGHTS = {
    "level3_code": 0.2,
    "level4_code": 0.3,
    "level5_code": 0.5,
}

# 相似度计算：各分量权重
SIMILARITY_WEIGHTS = {
    "ipc": 0.4,
    "brief": 0.4,
    "title": 0.2,
}

# 相似度筛选阈值及每个专利最多保留的近邻数
SIMILARITY_THRESHOLD = 0.75
TOP_K_NEIGHBORS = 1000

# 相似度计算批次大小（受 GPU 显存限制，可按需调大）
SIMILARITY_BATCH_SIZE = 300

# 相似度计算时每处理多少批次写一次 parquet 文件
SIMILARITY_SAVE_BATCH = 1000
