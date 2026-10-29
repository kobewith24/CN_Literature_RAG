# -*- coding: utf-8 -*-
"""RAG 评测：检索命中率 + 引用命中率。

对 qa_eval.jsonl 中每个问题：
  1) 调检索服务 top-10，检查 gold 文档是否在 top-5 / top-10
  2) 取前 N 个问题走完整问答链路，检查 gold 文档是否出现在引用里
"""
import argparse
import json
import time

import requests

import config

RETRIEVAL = "http://127.0.0.1:8000"
API = "http://127.0.0.1:8080"
QA = config.DATA / "qa_eval.jsonl"
OUT = config.DATA / "rag_eval.json"


def parse_sse(resp):
    event, data = None, []
    for raw in resp.iter_lines(decode_unicode=False):
        if raw is None:
            continue
        line = raw.decode("utf-8", errors="replace")
        if line == "":
            if event:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].lstrip())
        else:
            data.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask-n", type=int, default=20, help="跑完整问答的问题数")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(str(QA), encoding="utf-8") if l.strip()]
    print("questions:", len(rows), flush=True)

    hit5 = hit10 = 0
    ret_times = []
    for r in rows:
        t0 = time.time()
        resp = requests.get(RETRIEVAL + "/retrieve",
                            params={"q": r["question"], "k": 10, "mode": "hybrid"}, timeout=120)
        resp.raise_for_status()
        hits = resp.json().get("hits") or []
        ret_times.append((time.time() - t0) * 1000)
        doc_ids = [h.get("doc_id") for h in hits]
        if r["doc_id"] in doc_ids[:5]:
            hit5 += 1
        if r["doc_id"] in doc_ids[:10]:
            hit10 += 1

    cite_hit = 0
    ans_times = []
    n_ask = 0
    for r in rows[:args.ask_n]:
        t0 = time.time()
        sources = []
        try:
            with requests.get(API + "/api/ask", params={"q": r["question"], "k": 5},
                              stream=True, timeout=300) as resp:
                resp.raise_for_status()
                for event, data in parse_sse(resp):
                    if event == "sources":
                        try:
                            sources = json.loads(data)
                        except json.JSONDecodeError:
                            pass
                    elif event == "done":
                        break
                    elif event == "error":
                        print("  ask error:", data[:100], flush=True)
                        break
        except Exception as e:  # noqa: BLE001
            print("  ask exception:", str(e)[:100], flush=True)
        ans_times.append(time.time() - t0)
        n_ask += 1
        if any(s.get("doc_id") == r["doc_id"] for s in sources):
            cite_hit += 1

    result = {
        "questions": len(rows),
        "retrieval_hit@5": round(hit5 / len(rows), 4),
        "retrieval_hit@10": round(hit10 / len(rows), 4),
        "avg_retrieval_ms": round(sum(ret_times) / len(ret_times), 1),
        "ask_samples": n_ask,
        "citation_hit@5": round(cite_hit / max(1, n_ask), 4),
        "avg_answer_seconds": round(sum(ans_times) / max(1, len(ans_times)), 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
