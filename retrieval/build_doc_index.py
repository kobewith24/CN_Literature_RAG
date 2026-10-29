# -*- coding: utf-8 -*-
"""把 CSL（39 万余篇中文论文摘要）建成 passage 级 HNSW 索引。

产物:
  retrieval/data/doc_index_1024.bin   索引（id 与 doc_meta.jsonl 行号一致）
  retrieval/data/doc_meta.jsonl       每行一篇：doc_id/title/keywords/category/discipline

用法:
  python build_doc_index.py            # 全量
  python build_doc_index.py --limit 1000   # 冒烟测试
"""
import argparse
import gzip
import json
import time

import numpy as np

import config
from embed_lib import build_hnsw, encode, load_encoder


def load_docs(limit=None):
    docs = []
    t0 = time.time()
    with gzip.open(str(config.CSL_GZ), "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            title = (obj.get("title") or "").strip()
            abstract = (obj.get("abstract") or "").strip()
            if not abstract:
                continue
            docs.append({
                "doc_id": obj.get("doc_id") or "",
                "title": title,
                "abstract": abstract,
                "keywords": obj.get("keywords") or [],
                "category": obj.get("category") or "",
                "discipline": obj.get("discipline") or "",
            })
            if limit and len(docs) >= limit:
                break
            if len(docs) % 50000 == 0:
                print("  load %d (%.0fs)" % (len(docs), time.time() - t0), flush=True)
    return docs


def doc_text(d):
    kw = "，".join(d["keywords"]) if d["keywords"] else ""
    parts = [d["title"], d["abstract"]]
    if kw:
        parts.append("关键词：" + kw)
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    docs = load_docs(args.limit or None)
    print("docs loaded: %d (%.0fs)" % (len(docs), time.time() - t0), flush=True)

    texts = [doc_text(d) for d in docs]
    emb_cache = config.DATA / "doc_emb.npy"
    if emb_cache.exists():
        emb = np.load(str(emb_cache))
        print("loaded embedding cache: %s" % (emb.shape,), flush=True)
    else:
        encoder = load_encoder(max_seq_length=config.DOC_MAX_LEN, half=True)
        print("device: %s | max_len: %d | fp16" % (encoder.device, config.DOC_MAX_LEN), flush=True)
        emb = encode(encoder, texts, batch_size=96, max_length=config.DOC_MAX_LEN)
        np.save(str(emb_cache), emb)
        print("embeddings: %s (%.1f min)" % (emb.shape, (time.time() - t0) / 60), flush=True)

    build_hnsw(emb, config.DOC_INDEX)

    with open(str(config.DOC_META), "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps({"idx": i, **d}, ensure_ascii=False) + "\n")
    print("meta saved: %s (%.1f MB)" % (
        config.DOC_META, config.DOC_META.stat().st_size / 1048576.0), flush=True)
    print("ALL DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
