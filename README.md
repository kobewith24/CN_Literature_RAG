# 基于预训练语言模型的搜索排序系统（Search_Sys）

面向文献检索场景的小型搜索引擎，采用「语义召回 + 精排」两阶段架构，性能优于传统搜索排序算法。

## 项目简介

- **召回阶段**：基于 ERNIE-1.0 构建双塔语义索引模型，采用 in-batch negatives 对比学习，将 query 与文献标题编码为 256 维语义向量；基于 hnswlib 构建 HNSW 近似最近邻索引，实现毫秒级召回 top50，**recall@20 达 0.813**。
- **排序阶段**：基于 ERNIE-3.0-medium 以 margin ranking loss 训练 pairwise 排序模型，对召回结果进行精细排序，**排序模型 AUC 达 0.773**。
- **检索系统**：串联召回与排序模块，实现完整的「query → 语义召回 → 精排 → 返回结果」检索流程。

## 技术栈

Python · PaddlePaddle · PaddleNLP · hnswlib

## 目录结构

```
├── recall/                    # 召回模块
│   ├── finetune.py/.ipynb     #   ERNIE-1.0 双塔模型微调（in-batch negatives 对比学习）
│   ├── recall.py/.ipynb       #   对测试集 query 进行语义召回
│   ├── evaluate.py/.ipynb     #   计算 recall@N（N=1/5/10/20/50）
│   ├── ann_util.py            #   HNSW 索引构建工具
│   ├── base_model.py          #   双塔语义索引基础模型
│   ├── data.py / data2.py     #   数据集读取与样例转换
│   ├── model_param/           #   召回模型参数目录（权重文件见该目录 README）
│   └── recall_dataset/        #   召回阶段数据集
├── rank/                      # 排序模块
│   ├── train_pairwise.py/.ipynb   # 训练 pairwise 排序模型（margin ranking loss）并评估
│   ├── predict_pairwise.py/.ipynb # 利用训练好的模型对测试集进行排序
│   ├── model.py / data.py     #   排序模型定义与数据处理
│   ├── model_param/           #   排序模型参数目录（权重文件见该目录 README）
│   └── rank_dataset/          #   排序阶段数据集
└── search_system/             # 完整检索系统（召回 + 排序串联）
    ├── search.py/.ipynb       #   端到端检索入口
    ├── model.py / base_model.py / data.py / data2.py / ann_util.py
    └── （运行所需模型权重与 HNSW 索引文件较大，未上传至仓库，
        可分别使用 recall、rank 模块训练生成，或从本地备份拷贝）
```

## 运行步骤

### 1. 环境配置

```bash
pip install paddlepaddle paddlenlp hnswlib numpy
```

### 2. 召回模块（recall/）

1. `finetune.py`：利用 literature_search_data 数据集对 ERNIE-1.0 双塔模型进行微调；
2. `recall.py`：对测试集 query 进行语义召回（构建/加载 HNSW 索引，召回 top50）；
3. `evaluate.py`：对召回结果计算 recall@N，生成 result.tsv（记录 recall@1/5/10/20/50）。

### 3. 排序模块（rank/）

1. `train_pairwise.py`：训练 pairwise 排序模型，并在最后进行评估；
2. `predict_pairwise.py`：利用训练好的模型对测试集进行排序。

### 4. 完整检索系统（search_system/）

准备好召回模型权重（`recall_model_state.pdparams`）、排序模型权重（`rank_model_state.pdparams`）与 HNSW 索引文件（`final_index.bin`）后，运行 `search.py` 进行端到端检索。

> 说明：模型权重（*.pdparams）与索引文件（*.bin）单文件体积超过 GitHub 限制，未纳入版本管理，请按上述步骤自行训练生成，或参见各 `model_param/README.md`。

## 效果指标

| 指标 | 数值 |
|------|------|
| 召回 recall@20 | 0.813 |
| 排序模型 AUC | 0.773 |

## 备注

本项目为独立个人学习/开发项目，代码同时提供 `.py` 脚本与 `.ipynb` notebook 两种形式，内容一致，任选其一运行即可。
