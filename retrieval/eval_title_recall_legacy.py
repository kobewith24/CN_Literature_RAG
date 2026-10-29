# -*- coding: utf-8 -*-
"""用旧 ERNIE-1.0 + 旧 HNSW 索引，在同一 dev.csv 上重算 recall@N（基线）。

需在 conda `search` 环境（PaddlePaddle）中运行：
  D:\\Anaconda\\envs\\search\\python.exe eval_title_recall_legacy.py
"""
import json
import re
import sys
import time
from functools import partial

import hnswlib
import numpy as np
import paddle

SYS_DIR = r"D:\Projects\Search_Rank_System_1029\search_system"
sys.path.insert(0, SYS_DIR)

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer

from base_model import SemanticIndexBase
from data2 import convert_example

DEV = r"D:\Projects\Search_Rank_System_1029\recall\recall_dataset\dev.csv"
CORPUS = SYS_DIR + r"\corpus.csv"
INDEX = SYS_DIR + r"\final_index.bin"
WEIGHTS = SYS_DIR + r"\recall_model_state.pdparams"
OUT = r"D:\Projects\Search_Rank_System_1029\retrieval\data\metrics_title_ernie_legacy.json"

paddle.set_device("gpu")


def norm_text(s):
    return re.sub(r"[\s\W_]+", "", (s or "").lower(), flags=re.UNICODE)


def load_pairs(path):
    pairs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def main():
    t0 = time.time()
    corpus = [l.rstrip("\n") for l in open(CORPUS, encoding="utf-8", errors="replace") if l.strip()]
    print("corpus: %d" % len(corpus), flush=True)

    index = hnswlib.Index(space="ip", dim=256)
    index.load_index(INDEX)
    index.set_ef(128)
    print("index: %d" % index.get_current_count(), flush=True)

    pairs = load_pairs(DEV)
    queries = sorted({q for q, _ in pairs})
    print("pairs %d | queries %d" % (len(pairs), len(queries)), flush=True)

    tokenizer = AutoTokenizer.from_pretrained("ernie-1.0")
    pretrained = AutoModel.from_pretrained("ernie-1.0")
    model = SemanticIndexBase(pretrained, output_emb_size=256)
    model.set_dict(paddle.load(WEIGHTS))
    model.eval()

    def batchify(samples):
        fn = Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
        )
        return list(fn(samples))

    trans = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)

    emb = []
    bs = 64
    for start in range(0, len(queries), bs):
        chunk = queries[start:start + bs]
        ds = MapDataset([{"text": q} for q in chunk])
        sampler = paddle.io.BatchSampler(ds, batch_size=bs, shuffle=False)
        loader = paddle.io.DataLoader(
            dataset=ds.map(trans), batch_sampler=sampler, collate_fn=batchify, return_list=True
        )
        for batch in model.get_semantic_embedding(loader):
            emb.append(batch.numpy())
        if start % 3200 == 0:
            print("  encode %d/%d (%.0fs)" % (start + len(chunk), len(queries), time.time() - t0), flush=True)
    emb = np.concatenate(emb, axis=0)
    print("embeddings %s" % (emb.shape,), flush=True)

    recalled = [None] * len(queries)
    for start in range(0, len(queries), 512):
        end = min(len(queries), start + 512)
        idx, _ = index.knn_query(emb[start:end], k=50)
        for i in range(end - start):
            recalled[start + i] = [norm_text(corpus[j]) for j in idx[i]]

    hits = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    q2i = {q: i for i, q in enumerate(queries)}
    for q, target in pairs:
        want = norm_text(target)
        cand = recalled[q2i[q]]
        for k in hits:
            if want in cand[:k]:
                hits[k] += 1

    result = {
        "method": "ERNIE-1.0 dual-tower + hnswlib (legacy, recomputed)",
        "pairs": len(pairs),
        "recall@1": round(hits[1] / len(pairs), 4),
        "recall@5": round(hits[5] / len(pairs), 4),
        "recall@10": round(hits[10] / len(pairs), 4),
        "recall@20": round(hits[20] / len(pairs), 4),
        "recall@50": round(hits[50] / len(pairs), 4),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    open(OUT, "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
