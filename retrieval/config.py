# -*- coding: utf-8 -*-
"""全局配置：路径、模型位置、索引参数。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # retrieval/
PROJ = ROOT.parent                              # CN_Literature_RAG/
SYS = PROJ / "search_system"                    # 原检索系统（基线数据与权重）
DATA = ROOT / "data"                            # 新索引与处理中间产物

# 原项目资源（只读）
CORPUS_CSV = SYS / "corpus.csv"                 # 30 万条 标题+关键词
DEV_CSV = PROJ / "recall" / "recall_dataset" / "dev.csv"          # 20000 对 query-title
TEST_PAIRWISE_CSV = PROJ / "rank" / "rank_dataset" / "test_pairwise.csv"  # 1000 对带分数

# 模型
MODEL_BGE_M3 = Path(r"D:\models\bge-m3")
MODEL_RERANKER = Path(r"D:\models\bge-reranker-v2-m3")

# 索引
EMB_DIM = 1024
TITLE_INDEX = DATA / "title_index_1024.bin"
DOC_INDEX = DATA / "doc_index_1024.bin"
DOC_META = DATA / "doc_meta.jsonl"
CSL_GZ = DATA / "csl.jsonl.gz"

# 检索参数
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
DEFAULT_TOPK = 50

# 编码参数
TITLE_MAX_LEN = 96
DOC_MAX_LEN = 384
ENCODE_BATCH = 64

# CPU 线程上限：hnswlib 建索引是 CPU 密集型；默认 4 线程，
# 可用环境变量 RAG_THREADS 覆盖（如 RAG_THREADS=8）。
import os  # noqa: E402

HNSW_BUILD_THREADS = int(os.environ.get("RAG_THREADS", "4"))
