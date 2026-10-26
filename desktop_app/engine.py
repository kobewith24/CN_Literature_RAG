# -*- coding: utf-8 -*-
"""检索引擎：加载 ERNIE 双塔召回模型 + pairwise 精排模型，提供 search() 接口。

与 search_system/search.py 的流程一致：
  query -> 语义向量(ERNIE-1.0) -> HNSW 召回 top50 -> ERNIE-3.0 精排
"""
import os
import sys
import threading
import time

import hnswlib
import paddle
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
SYS_DIR = os.path.join(PROJ_DIR, "search_system")
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from base_model import SemanticIndexBase  # noqa: E402
from data import convert_pairwise_example  # noqa: E402
from data2 import convert_example, gen_id2corpus  # noqa: E402
from model import PairwiseMatching  # noqa: E402

paddle.set_device("cpu")

OUTPUT_EMB_SIZE = 256
RECALL_MAX_LEN = 60
RANK_MAX_LEN = 64


def _make_loader(dataset, batchify_fn, trans_fn, batch_size):
    ds = dataset.map(trans_fn) if trans_fn else dataset
    sampler = paddle.io.BatchSampler(ds, batch_size=batch_size, shuffle=False)
    return paddle.io.DataLoader(
        ds, batch_sampler=sampler, collate_fn=batchify_fn, return_list=True
    )


class SearchEngine(object):
    """两阶段检索：语义召回 + 精排。"""

    def __init__(self):
        self.lock = threading.Lock()
        t0 = time.perf_counter()

        # ---- 召回：ERNIE-1.0 双塔语义索引 ----
        self.recall_tokenizer = AutoTokenizer.from_pretrained("ernie-1.0")
        recall_ptm = AutoModel.from_pretrained("ernie-1.0")
        self.recall_model = SemanticIndexBase(recall_ptm, output_emb_size=OUTPUT_EMB_SIZE)
        self.recall_model.set_dict(
            paddle.load(os.path.join(SYS_DIR, "recall_model_state.pdparams"))
        )
        self.recall_model.eval()

        # ---- 语料库 + HNSW 索引 ----
        self.id2corpus = gen_id2corpus(os.path.join(SYS_DIR, "corpus.csv"))
        self.index = hnswlib.Index(space="ip", dim=OUTPUT_EMB_SIZE)
        self.index.load_index(os.path.join(SYS_DIR, "final_index.bin"))
        self.index.set_ef(100)

        # ---- 精排：ERNIE-3.0-medium pairwise ----
        self.rank_tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
        rank_ptm = AutoModel.from_pretrained("ernie-3.0-medium-zh")
        self.rank_model = PairwiseMatching(rank_ptm)
        self.rank_model.set_dict(
            paddle.load(os.path.join(SYS_DIR, "rank_model_state.pdparams"))
        )
        self.rank_model.eval()

        self.load_seconds = time.perf_counter() - t0
        self.corpus_size = len(self.id2corpus)

    # ------------------------------------------------------------------
    def _recall_batchify(self, samples):
        fn = Tuple(
            Pad(axis=0, pad_val=self.recall_tokenizer.pad_token_id, dtype="int64"),
            Pad(axis=0, pad_val=self.recall_tokenizer.pad_token_type_id, dtype="int64"),
        )
        return list(fn(samples))

    def _rank_batchify(self, samples):
        fn = Tuple(
            Pad(axis=0, pad_val=self.rank_tokenizer.pad_token_id, dtype="int64"),
            Pad(axis=0, pad_val=self.rank_tokenizer.pad_token_type_id, dtype="int64"),
        )
        return list(fn(samples))

    @paddle.no_grad()
    def _rank_predict(self, loader):
        probs = []
        for input_ids, token_type_ids in loader:
            batch_prob = self.rank_model.predict(
                input_ids=input_ids, token_type_ids=token_type_ids
            ).numpy()
            probs.append(batch_prob)
        import numpy as np

        return np.concatenate(probs, axis=0)

    # ------------------------------------------------------------------
    def search(self, query, topk=50):
        """返回 {query, recall: [...], ranked: [...], elapsed}。"""
        query = (query or "").strip()
        if not query:
            raise ValueError("查询词为空")

        with self.lock:
            t0 = time.perf_counter()

            # 1) 召回
            query_ds = MapDataset([{"text": query}])
            recall_loader = _make_loader(
                query_ds,
                self._recall_batchify,
                lambda x: convert_example(
                    x, tokenizer=self.recall_tokenizer, max_seq_length=RECALL_MAX_LEN
                ),
                batch_size=1,
            )
            embedding = next(self.recall_model.get_semantic_embedding(recall_loader)).numpy()
            recalled_idx, cosine_sims = self.index.knn_query(embedding, topk)

            recall_items = []
            for idx, doc_idx in enumerate(recalled_idx[0]):
                title = self.id2corpus[int(doc_idx)]
                recall_items.append(
                    {"title": title, "score": float(1.0 - cosine_sims[0][idx])}
                )

            # 2) 精排
            rank_ds = MapDataset(
                [{"query": query, "title": it["title"]} for it in recall_items]
            )
            rank_loader = _make_loader(
                rank_ds,
                self._rank_batchify,
                lambda x: convert_pairwise_example(
                    x,
                    tokenizer=self.rank_tokenizer,
                    max_seq_length=RANK_MAX_LEN,
                    phase="predict",
                ),
                batch_size=16,
            )
            probs = self._rank_predict(rank_loader)

        ranked = [
            {
                "title": it["title"],
                "score": float(prob[0]),
                "recall_score": it["score"],
            }
            for it, prob in zip(recall_items, probs)
        ]
        ranked.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "recall": recall_items,
            "ranked": ranked,
            "elapsed": time.perf_counter() - t0,
        }
