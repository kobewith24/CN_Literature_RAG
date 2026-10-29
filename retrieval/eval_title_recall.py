# -*- coding: utf-8 -*-
"""在 dev.csv（20000 对 query-title）上评估 BGE-M3 新索引的 recall@N。

与旧项目 evaluate.py 的口径一致：检索 top-K，命中目标标题即算召回。
产物: docs/metrics.json 中的 bge_m3_title 部分。
"""
import json
import time

import hnswlib

import config
from embed_lib import encode, load_encoder, norm_text, read_lines


def load_pairs():
    pairs = []
    for line in read_lines(config.DEV_CSV):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def main():
    t0 = time.time()
    corpus = read_lines(config.CORPUS_CSV)
    print("corpus: %d lines" % len(corpus), flush=True)

    index = hnswlib.Index(space="ip", dim=config.EMB_DIM)
    index.load_index(str(config.TITLE_INDEX))
    index.set_ef(config.HNSW_EF_SEARCH)
    print("index loaded: %d elements" % index.get_current_count(), flush=True)

    pairs = load_pairs()
    queries = sorted({q for q, _ in pairs})
    print("pairs: %d | unique queries: %d" % (len(pairs), len(queries)), flush=True)

    encoder = load_encoder(max_seq_length=config.TITLE_MAX_LEN)
    q_emb = encode(encoder, queries, max_length=config.TITLE_MAX_LEN)
    q2idx = {q: i for i, q in enumerate(queries)}

    topk = config.DEFAULT_TOPK
    nq = len(queries)
    recalled = [None] * nq
    batch = 512
    knn_seconds = 0.0
    for start in range(0, nq, batch):
        end = min(nq, start + batch)
        _t = time.time()
        idx, _ = index.knn_query(q_emb[start:end], k=topk)
        knn_seconds += time.time() - _t
        for i in range(end - start):
            recalled[start + i] = [norm_text(corpus[j]) for j in idx[i]]
        if end % 4096 == 0:
            print("  query %d/%d" % (end, nq), flush=True)

    hits = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    for q, target in pairs:
        want = norm_text(target)
        cand = recalled[q2idx[q]]
        for k in hits:
            if want in cand[:k]:
                hits[k] += 1
    latency_ms = knn_seconds / nq * 1000

    result = {
        "method": "bge-m3 + hnswlib (title corpus, 300k)",
        "pairs": len(pairs),
        "recall@1": round(hits[1] / len(pairs), 4),
        "recall@5": round(hits[5] / len(pairs), 4),
        "recall@10": round(hits[10] / len(pairs), 4),
        "recall@20": round(hits[20] / len(pairs), 4),
        "recall@50": round(hits[50] / len(pairs), 4),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "avg_retrieval_ms": round(latency_ms, 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    out = config.DATA / "metrics_title_bge_m3.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out, flush=True)


if __name__ == "__main__":
    main()
