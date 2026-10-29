# -*- coding: utf-8 -*-
"""摘要索引质量抽检：用几个查询看 top-5 命中的标题与分数。"""
import json
import sys

import hnswlib

import config
from embed_lib import load_encoder

QUERIES = [
    "双目立体视觉的相机标定方法",
    "基于深度学习的图像识别",
    "光场三维显示的关键技术",
    "锂离子电池的容量衰减机理",
]


def read_meta_at(f, offset):
    f.seek(offset)
    return json.loads(f.readline().decode("utf-8"))


def main():
    index = hnswlib.Index(space="ip", dim=config.EMB_DIM)
    index.load_index(str(config.DOC_INDEX))
    index.set_ef(config.HNSW_EF_SEARCH)
    print("index docs:", index.get_current_count())

    # 建立行偏移
    offsets = []
    with open(str(config.DOC_META), "rb") as f:
        pos = 0
        for line in f:
            offsets.append(pos)
            pos += len(line)
    print("meta lines:", len(offsets))

    enc = load_encoder(max_seq_length=config.DOC_MAX_LEN)
    print("device:", enc.device)
    emb = enc.encode(QUERIES, batch_size=len(QUERIES), convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)

    labels, dists = index.knn_query(emb, k=5)
    with open(str(config.DOC_META), "rb") as f:
        for i, q in enumerate(QUERIES):
            print("\nQ: %s" % q)
            for rank, lab in enumerate(labels[i]):
                meta = read_meta_at(f, offsets[int(lab)])
                print("  %.4f  [%s] %s" % (1.0 - dists[i][rank], meta.get("category", ""), meta.get("title", "")[:60]))


if __name__ == "__main__":
    main()
