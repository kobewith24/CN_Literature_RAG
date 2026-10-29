# -*- coding: utf-8 -*-
"""在 rank/test_pairwise.csv（1000 对，带连续相关度分）上评估 bge-reranker-v2-m3。

原项目报的 AUC=0.773 是基于 pairwise 模型；这里用同一批数据重算新模型指标，
并同时给出 Spearman 相关（不依赖二值化阈值）。
"""
import json
import time

import numpy as np

import config
from embed_lib import load_reranker, read_lines


def auc_score(labels, scores):
    """基于秩的 AUC，labels ∈ {0,1}。"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def main():
    t0 = time.time()
    rows = []
    for line in read_lines(config.TEST_PAIRWISE_CSV):
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                rows.append((parts[0].strip(), parts[1].strip(), float(parts[2])))
            except ValueError:
                continue
    print("test pairs: %d" % len(rows), flush=True)

    reranker = load_reranker()
    print("device: %s" % reranker.device, flush=True)
    scores = reranker.predict([(q, t) for q, t, _ in rows], batch_size=32, show_progress_bar=True)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    gold = np.asarray([s for _, _, s in rows], dtype=float)
    # 注意：test_pairwise.csv 第三列是「距离/不相似度」——越小越相关
    relevance = -gold

    med = float(np.median(gold))
    labels_med = (gold <= med).astype(int)
    labels_07 = (gold <= 0.7).astype(int)
    result = {
        "method": "bge-reranker-v2-m3 (cross-encoder)",
        "pairs": len(rows),
        "note": "gold 列为距离（越小越相关），已按此方向计算",
        "auc_median_split": round(auc_score(labels_med, scores), 4),
        "auc_threshold_0.7": round(auc_score(labels_07, scores), 4),
        "spearman": round(spearman(relevance, scores), 4),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    out = config.DATA / "metrics_rank_bge_reranker.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out, flush=True)


if __name__ == "__main__":
    main()
