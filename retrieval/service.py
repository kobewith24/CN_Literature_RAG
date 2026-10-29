# -*- coding: utf-8 -*-
"""检索服务（FastAPI）：标题索引 + 摘要索引双路召回，reranker 统一重排。

启动:
  python service.py                 # 127.0.0.1:8000
接口:
  GET  /health
  GET  /retrieve?q=深度学习&k=10&mode=hybrid
"""
import io
import json
import os
import time

import hnswlib
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# CPU 线程上限（默认 4），避免检索服务占满 CPU
_THREADS = int(os.environ.get("RAG_THREADS", "4"))
torch.set_num_threads(_THREADS)

import config
from embed_lib import load_encoder, load_reranker

app = FastAPI(title="Search_Rank_System RAG Retrieval Service")

STATE = {}


# --------------------------------------------------------------------- 加载
def _load_meta_offsets(path):
    """为 doc_meta.jsonl 建行偏移索引，避免把 40 万条摘要读进内存。"""
    offsets = []
    with io.open(str(path), "rb") as f:
        pos = 0
        for line in f:
            offsets.append(pos)
            pos += len(line)
    return offsets


def _read_meta_line(path, offset):
    with io.open(str(path), "rb") as f:
        f.seek(offset)
        return json.loads(f.readline().decode("utf-8"))


def _load_index(path, dim):
    idx = hnswlib.Index(space="ip", dim=dim)
    idx.load_index(str(path))
    idx.set_ef(config.HNSW_EF_SEARCH)
    return idx


@app.on_event("startup")
def startup():
    t0 = time.time()
    STATE["encoder"] = load_encoder(max_seq_length=config.DOC_MAX_LEN, half=True)
    STATE["reranker"] = load_reranker()
    STATE["device"] = str(STATE["encoder"].device)
    if not STATE["device"].startswith("cuda"):
        raise RuntimeError("必须使用 GPU 推理：CUDA 不可用")

    print("encoder device=%s | reranker device=%s" % (
        STATE["encoder"].device,
        next(STATE["reranker"].model.parameters()).device), flush=True)

    STATE["corpus"] = [
        line.rstrip("\n")
        for line in io.open(str(config.CORPUS_CSV), encoding="utf-8", errors="replace")
        if line.strip()
    ]
    STATE["title_index"] = _load_index(config.TITLE_INDEX, config.EMB_DIM)

    STATE["doc_index"] = None
    STATE["doc_offsets"] = None
    if config.DOC_INDEX.exists() and config.DOC_META.exists():
        STATE["doc_index"] = _load_index(config.DOC_INDEX, config.EMB_DIM)
        STATE["doc_offsets"] = _load_meta_offsets(config.DOC_META)

    STATE["ready"] = True
    STATE["load_seconds"] = time.time() - t0
    print("service ready in %.1fs | titles=%d | docs=%s" % (
        STATE["load_seconds"], len(STATE["corpus"]),
        STATE["doc_index"].get_current_count() if STATE["doc_index"] is not None else 0), flush=True)


# --------------------------------------------------------------------- 检索
def _encode_query(q):
    enc = STATE["encoder"]
    return enc.encode([q], batch_size=1, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)


def _title_hits(qvec, topk):
    idx = STATE["title_index"]
    labels, dists = idx.knn_query(qvec, k=topk, num_threads=_THREADS)
    hits = []
    for i, lab in enumerate(labels[0]):
        hits.append({
            "source": "title",
            "score": float(1.0 - dists[0][i]),
            "title": STATE["corpus"][int(lab)],
            "text": "",
            "doc_id": "",
        })
    return hits


def _doc_hits(qvec, topk):
    if STATE["doc_index"] is None:
        return []
    labels, dists = STATE["doc_index"].knn_query(qvec, k=topk, num_threads=_THREADS)
    hits = []
    for i, lab in enumerate(labels[0]):
        meta = _read_meta_line(config.DOC_META, STATE["doc_offsets"][int(lab)])
        hits.append({
            "source": "doc",
            "score": float(1.0 - dists[0][i]),
            "title": meta.get("title", ""),
            "text": meta.get("abstract", ""),
            "doc_id": meta.get("doc_id", ""),
            "category": meta.get("category", ""),
            "discipline": meta.get("discipline", ""),
            "keywords": meta.get("keywords", []),
        })
    return hits


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    elapsed_ms: float
    hits: list


@app.get("/health")
def health():
    return {
        "status": "ok" if STATE.get("ready") else "loading",
        "device": STATE.get("device"),
        "titles": len(STATE.get("corpus", [])),
        "docs": STATE["doc_index"].get_current_count() if STATE.get("doc_index") is not None else 0,
        "load_seconds": round(STATE.get("load_seconds", 0), 1),
    }


@app.get("/retrieve", response_model=RetrieveResponse)
def retrieve(
    q: str = Query(..., min_length=1, description="查询词/标题/摘要/问题"),
    k: int = Query(10, ge=1, le=100),
    mode: str = Query("hybrid", pattern="^(hybrid|title|doc)$"),
    candidates: int = Query(50, ge=10, le=200),
):
    if not STATE.get("ready"):
        raise HTTPException(status_code=503, detail="service loading")

    t0 = time.time()
    qvec = _encode_query(q)

    hits = []
    if mode in ("title", "hybrid"):
        hits += _title_hits(qvec, candidates)
    if mode in ("doc", "hybrid"):
        hits += _doc_hits(qvec, candidates)

    if mode == "hybrid" and hits and STATE["reranker"] is not None:
        pairs = [(q, h["text"] or h["title"]) for h in hits]
        scores = STATE["reranker"].predict(pairs, batch_size=32, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        hits.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
    else:
        hits.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": q,
        "mode": mode,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "hits": hits[:k],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
