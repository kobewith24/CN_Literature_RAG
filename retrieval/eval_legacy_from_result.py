# -*- coding: utf-8 -*-
"""用旧管线已产出的 recall_result.txt（20k 查询 × top50）计算 ERNIE 基线 recall@N。

与 eval_title_recall.py 使用完全相同的匹配口径（去空白标点后精确匹配），保证可比。
"""
import io
import json
import re

BASE = r"D:\Projects\Search_Rank_System"
DEV = BASE + r"\recall\recall_dataset\dev.csv"
RESULT = BASE + r"\recall\recall_result_file\recall_result.txt"
OUT = r"D:\Projects\Search_Rank_System_1029\retrieval\data\metrics_title_ernie_legacy.json"


def norm(s):
    return re.sub(r"[\s\W_]+", "", (s or "").lower(), flags=re.UNICODE)


def main():
    pairs = []
    for line in io.open(DEV, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip()))
    print("dev pairs:", len(pairs))

    # 每 50 行对应一条 dev query 的召回结果
    groups = []
    cur = []
    n = 0
    with io.open(RESULT, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            cur.append(norm(parts[1]))
            n += 1
            if len(cur) == 50:
                groups.append(cur)
                cur = []
    if cur:
        groups.append(cur)
    print("result lines: %d | groups: %d" % (n, len(groups)))

    m = min(len(groups), len(pairs))
    hits = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    for i in range(m):
        want = norm(pairs[i][1])
        cand = groups[i]
        for k in hits:
            if want in cand[:k]:
                hits[k] += 1

    result = {
        "method": "ERNIE-1.0 dual-tower + HNSW (legacy recall_result.txt)",
        "pairs": m,
        "recall@1": round(hits[1] / m, 4),
        "recall@5": round(hits[5] / m, 4),
        "recall@10": round(hits[10] / m, 4),
        "recall@20": round(hits[20] / m, 4),
        "recall@50": round(hits[50] / m, 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
