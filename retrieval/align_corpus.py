# -*- coding: utf-8 -*-
"""统计原 30 万题录与 CSL 39.6 万摘要语料的对齐率（低 CPU 占用）。

匹配规则：
1) 严格：归一化后「标题+关键词」完全一致
2) 前缀：CSL 标题（归一化）是题录行（归一化）的前缀——题录行 = 标题+关键词
"""
import io
import json
import re
import time
from collections import defaultdict

import config

OUT = config.DATA / "alignment_stats.json"


def norm(s):
    return re.sub(r"[\s\W_]+", "", (s or "").lower(), flags=re.UNICODE)


def main():
    t0 = time.time()

    # 1) 读取 CSL 标题（流式，避免把摘要读入内存）
    exact = {}
    by_prefix = defaultdict(list)
    n_doc = 0
    with io.open(str(config.DOC_META), "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            title = norm(obj.get("title"))
            if not title:
                continue
            key = norm(obj.get("title", "") + "".join(obj.get("keywords") or []))
            exact[key] = obj.get("doc_id")
            by_prefix[title[:8]].append((title, obj.get("doc_id")))
            n_doc += 1
    print("CSL docs: %d | prefix buckets: %d (%.0fs)" % (n_doc, len(by_prefix), time.time() - t0), flush=True)

    # 2) 逐条匹配题录
    total = 0
    strict_hits = 0
    prefix_hits = 0
    strict_only = 0
    examples = []
    with io.open(str(config.CORPUS_CSV), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            ln = norm(line)
            if ln in exact:
                strict_hits += 1
                prefix_hits += 1
                if len(examples) < 5:
                    examples.append({"corpus": line[:80], "doc_id": exact[ln], "match": "strict"})
                continue
            matched = None
            for title, doc_id in by_prefix.get(ln[:8], ()):
                if len(title) <= len(ln) and ln.startswith(title):
                    matched = doc_id
                    break
            if matched:
                prefix_hits += 1
                if len(examples) < 5:
                    examples.append({"corpus": line[:80], "doc_id": matched, "match": "prefix"})

    result = {
        "corpus_lines": total,
        "csl_docs": n_doc,
        "strict_match": strict_hits,
        "strict_rate": round(strict_hits / max(1, total), 4),
        "prefix_match": prefix_hits,
        "prefix_rate": round(prefix_hits / max(1, total), 4),
        "examples": examples,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    io.open(str(OUT), "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
