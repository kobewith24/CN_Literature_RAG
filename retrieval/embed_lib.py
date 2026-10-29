# -*- coding: utf-8 -*-
"""共享工具：模型加载、批量编码、索引构建、文本读取。"""
import io
import json
import os
import time

import numpy as np

from config import (
    EMB_DIM,
    ENCODE_BATCH,
    HNSW_BUILD_THREADS,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    MODEL_BGE_M3,
    MODEL_RERANKER,
)


def torch_device():
    """强制 GPU：没有可用 CUDA 时直接报错（除非显式设置 ALLOW_CPU=1）。"""
    import os

    import torch

    # 限制 CPU 线程，避免任何 CPU 侧算子占满 32 核
    torch.set_num_threads(int(os.environ.get("RAG_THREADS", "4")))

    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("ALLOW_CPU") == "1":
        print("[WARN] CUDA 不可用，按 ALLOW_CPU=1 回退 CPU", flush=True)
        return "cpu"
    raise RuntimeError(
        "CUDA 不可用：本项目的向量化/重排必须使用 GPU。"
        "如确需 CPU 运行，请设置环境变量 ALLOW_CPU=1。")


def load_encoder(max_seq_length=512, half=False):
    """加载 BGE-M3（sentence-transformers 接口，dense 向量）。"""
    import torch
    from sentence_transformers import SentenceTransformer

    device = torch_device()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = SentenceTransformer(str(MODEL_BGE_M3), device=device)
    model.max_seq_length = max_seq_length
    if half and device == "cuda":
        model.half()
    return model


def load_reranker():
    from sentence_transformers import CrossEncoder

    device = torch_device()
    model = CrossEncoder(str(MODEL_RERANKER), device=device)
    if device == "cuda":
        model.model.half()
    return model


def encode(encoder, texts, batch_size=ENCODE_BATCH, max_length=None, normalize=True):
    """批量编码，返回 float32 矩阵 [N, 1024]。"""
    if max_length:
        encoder.max_seq_length = max_length
    emb = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(emb, dtype="float32")


def build_hnsw(embeddings, path, dim=EMB_DIM, m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION):
    import hnswlib

    n = len(embeddings)
    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    t0 = time.time()
    step = 20000
    for start in range(0, n, step):
        end = min(n, start + step)
        index.add_items(embeddings[start:end], np.arange(start, end),
                        num_threads=HNSW_BUILD_THREADS)
        print("  add_items %d/%d (%.0fs)" % (end, n, time.time() - t0), flush=True)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    index.save_index(str(path))
    print("saved index: %s (%.1f MB)" % (path, os.path.getsize(path) / 1048576.0), flush=True)
    return index


def read_lines(path):
    with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def norm_text(s):
    """比较用的归一化：去空白与标点，转小写。"""
    import re

    return re.sub(r"[\s\W_]+", "", (s or "").lower(), flags=re.UNICODE)
