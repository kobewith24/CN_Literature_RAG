# -*- coding: utf-8 -*-
"""生成 RAG 评测集：从 CSL 摘要抽样，让本地 Qwen3 为每篇写一个可由摘要回答的问题。

产物: retrieval/data/qa_eval.jsonl
用法: python build_qa_eval.py --n 60
"""
import argparse
import gzip
import json
import random
import re
import time

import requests

import config

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
OUT = config.DATA / "qa_eval.jsonl"

PROMPT = """请根据下面这篇论文的摘要，写一个中文问题：该问题必须能且只能由这段摘要回答。
只输出问题本身，不要编号、不要解释、不要引号。

标题：{title}
摘要：{abstract}
"""


def sample_docs(n):
    docs = []
    with gzip.open(str(config.CSL_GZ), "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            title = (obj.get("title") or "").strip()
            abstract = (obj.get("abstract") or "").strip()
            if 120 <= len(abstract) <= 400 and title:
                docs.append({
                    "doc_id": obj.get("doc_id"),
                    "title": title,
                    "abstract": abstract,
                    "category": obj.get("category", ""),
                })
    random.seed(2026)
    return random.sample(docs, min(n, len(docs)))


def gen_question(doc):
    prompt = PROMPT.format(title=doc["title"], abstract=doc["abstract"][:600])
    r = requests.post(OLLAMA + "/api/generate", json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 80, "num_thread": 4},
    }, timeout=180)
    r.raise_for_status()
    text = (r.json().get("response") or "").strip()
    text = re.sub(r"^(问题|问：|Q[:：])\s*", "", text)
    return text.split("\n")[0].strip().strip('"').strip("“”")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    docs = sample_docs(args.n)
    print("sampled docs:", len(docs), flush=True)
    ok = 0
    t0 = time.time()
    with open(str(OUT), "w", encoding="utf-8") as f:
        for i, d in enumerate(docs, 1):
            try:
                q = gen_question(d)
            except Exception as e:  # noqa: BLE001
                print("  err", i, str(e)[:60], flush=True)
                continue
            if not q:
                continue
            rec = {"qid": i, "question": q, **d}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
            if i % 10 == 0:
                print("  %d/%d (%.0fs)" % (i, len(docs), time.time() - t0), flush=True)
    print("saved %d questions -> %s (%.1f min)" % (ok, OUT, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
