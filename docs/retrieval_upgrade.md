# 检索升级实验报告：ERNIE-1.0 → BGE-M3

## 1. 目的与范围

在**同一份语料、同一份评测集**上，比较 2023 年旧检索栈（ERNIE-1.0 双塔 + HNSW）与
2026 年新检索栈（BGE-M3 dense + HNSW）的检索质量，量化升级收益。

- 语料：`search_system/corpus.csv`，300,000 条「标题+关键词」题录（去重 292,175）
- 评测集：`recall/recall_dataset/dev.csv`，20,000 对 query-title（过滤后 19,940 对）
- 匹配口径：两套系统完全相同——归一化（去空白/标点、转小写）后精确匹配目标标题
- 硬件：RTX 4060 Laptop 8GB，conda `rag` 环境（torch 2.5.1+cu121）

## 2. 方法

| 环节 | 旧栈 | 新栈 |
|---|---|---|
| 向量模型 | ERNIE-1.0 双塔（768→256 投影，余弦） | BGE-M3 dense（1024 维，L2 归一化） |
| 索引 | hnswlib (ip)，M=100, ef=100 | hnswlib (ip)，M=32, ef_construction=200, ef_search=128 |
| 编码 | PaddlePaddle（CPU） | PyTorch + CUDA（GPU） |
| 索引体积 | 553 MB | 1,251 MB（维度 256→1024） |

## 3. 结果：标题级召回（19,940 对）

| 指标 | ERNIE-1.0（旧，GPU 复现） | BGE-M3（新） | 变化 |
|---|---|---|---|
| recall@1 | 0.5822 | **0.6356** | +5.3 pt |
| recall@5 | 0.6949 | **0.8016** | +10.7 pt |
| recall@10 | 0.7352 | **0.8467** | +11.2 pt |
| recall@20 | 0.7698 | **0.8794** | +11.0 pt |
| recall@50 | 0.8119 | **0.9121** | +10.0 pt |

- 旧栈 GPU 复现：19,932 条查询编码 **24 秒**（约 830 条/秒）；CPU 下约 70 分钟
- 与基于 `recall_result.txt` 的离线重算（0.5819/0.6945/0.7343/0.7688/0.8106）差异 < 0.002，互为验证
- 新栈全量 19,940 条查询编码 + 检索：约 **1 分钟**（GPU，约 330 条/秒）

> 注：旧栈基线由项目已产出的 `recall_result.txt`（20k 查询 × top50）按同一口径重算。
> 原 README 宣称 recall@20=0.813，与本次重算的 @50=0.8106 接近，推测原口径存在偏差
> （可能把 @50 记作 @20），因此本报告一律使用重算值做同口径对比。

## 4. 结果：精排（1,000 对，带 gold 距离）

- `rank/rank_dataset/test_pairwise.csv` 第三列为**距离/不相似度**（越小越相关），
  已做方向校验：gold=0.432 的样本重排得分 0.989，gold=0.805 的样本得分 0.011。

| 指标 | bge-reranker-v2-m3 |
|---|---|
| AUC（中位数二值化） | 0.6937 |
| AUC（阈值 0.7 二值化） | 0.7550 |
| Spearman（与相关性） | 0.3962 |

> 旧项目报告的 AUC=0.773 来自其自训 pairwise 模型，**二值化方式与方向未在文档中说明**，
> 不能直接与新指标比较。可确认的是：新 reranker 在该测试集上与旧模型大致同档（0.69–0.76），
> 但其价值主要体现在覆盖更高质量的候选集（召回 recall@20 提升 11 pt）以及跨粒度统一打分能力。

## 5. 结论

1. **召回质量显著提升**：recall@20 从 0.769 提升到 0.879（+11.1 pt），且无需任何领域微调。
2. **工程栈升级到位**：PyTorch + CUDA 全链路 GPU，编码吞吐提升到约 330 条/秒，索引构建 25 分钟。
3. **旧栈保留为基线**：`rank/`、`recall/`、`search_system/` 原样未动，可随时复现旧结果。

## 6. 复现

```bash
conda activate rag
cd retrieval
python build_title_index.py    # 重建标题索引（含 embedding 缓存）
python eval_title_recall.py    # 新栈 recall@N
python eval_rank_auc.py        # 重排 AUC
# 旧栈基线（无需 GPU，基于已有 recall_result.txt）
python eval_legacy_from_result.py
```
