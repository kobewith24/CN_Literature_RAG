# -*- coding: utf-8 -*-
"""用 BGE-M3 重建 30 万条标题语料（原 corpus.csv）的 HNSW 索引。

编码结果缓存在 data/title_emb.npy，中断后可直接复用。
产物: retrieval/data/title_index_1024.bin（索引 id == corpus 行号）
"""
import time

import numpy as np

import config
from embed_lib import build_hnsw, encode, load_encoder, read_lines

EMB_CACHE = config.DATA / "title_emb.npy"


def main():
    t0 = time.time()
    lines = read_lines(config.CORPUS_CSV)
    print("corpus lines: %d" % len(lines), flush=True)

    if EMB_CACHE.exists():
        emb = np.load(str(EMB_CACHE))
        print("loaded embedding cache: %s (%.1f min)" % (emb.shape, (time.time() - t0) / 60), flush=True)
    else:
        encoder = load_encoder(max_seq_length=config.TITLE_MAX_LEN)
        print("device: %s" % encoder.device, flush=True)
        emb = encode(encoder, lines, max_length=config.TITLE_MAX_LEN)
        np.save(str(EMB_CACHE), emb)
        print("embeddings: %s | encoded in %.1f min" % (emb.shape, (time.time() - t0) / 60), flush=True)

    build_hnsw(emb, config.TITLE_INDEX)
    print("ALL DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
