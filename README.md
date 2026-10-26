# 基于预训练语言模型的搜索排序系统（Search_Rank_System）

面向文献检索场景的小型搜索引擎，采用「语义召回 + 精排」两阶段架构，性能优于传统搜索排序算法。

## 项目简介

- **召回阶段**：基于 ERNIE-1.0 构建双塔语义索引模型，采用 in-batch negatives 对比学习，将 query 与文献标题编码为 256 维语义向量；基于 hnswlib 构建 HNSW 近似最近邻索引，实现毫秒级召回 top50，**recall@20 达 0.813**。
- **排序阶段**：基于 ERNIE-3.0-medium 以 margin ranking loss 训练 pairwise 排序模型，对召回结果进行精细排序，**排序模型 AUC 达 0.773**。
- **检索系统**：串联召回与排序模块，实现完整的「query → 语义召回 → 精排 → 返回结果」检索流程。

## 技术栈

Python · PaddlePaddle · PaddleNLP · hnswlib

## 目录结构

```
├── docs/                      # 环境配置命令（本机 / GPU 服务器）
├── requirements.txt           # Python 依赖清单
├── recall/                    # 召回模块
│   ├── finetune.py/.ipynb     #   ERNIE-1.0 双塔模型微调（in-batch negatives 对比学习）
│   ├── recall.py/.ipynb       #   对测试集 query 进行语义召回
│   ├── evaluate.py/.ipynb     #   计算 recall@N（N=1/5/10/20/50）
│   ├── ann_util.py            #   HNSW 索引构建工具
│   ├── base_model.py          #   双塔语义索引基础模型
│   ├── data.py / data2.py     #   数据集读取与样例转换
│   ├── model_param/           #   召回模型参数目录（权重文件见该目录 README）
│   ├── recall_dataset/        #   召回阶段数据集（含语料库 corpus.csv）
│   └── recall_result_file/    #   召回结果与评估结果（本地生成，未上传）
├── rank/                      # 排序模块
│   ├── train_pairwise.py/.ipynb   # 训练 pairwise 排序模型（margin ranking loss）并评估
│   ├── predict_pairwise.py/.ipynb # 利用训练好的模型对测试集进行排序
│   ├── model.py / data.py     #   排序模型定义与数据处理
│   ├── model_param/           #   排序模型参数目录（权重文件见该目录 README）
│   └── rank_dataset/          #   排序阶段数据集（训练集 train_pairwise.csv 过大未上传）
├── search_system/             # 完整检索系统（召回 + 排序串联）
│   ├── search.py/.ipynb       #   端到端检索入口（命令行交互）
│   ├── model.py / base_model.py / data.py / data2.py / ann_util.py
│   └── （运行所需模型权重、HNSW 索引文件与语料库 corpus.csv 较大，
│       权重与索引未上传至仓库，可分别使用 recall、rank 模块训练生成，或从本地备份拷贝）
└── desktop_app/               # 桌面客户端（Tkinter 图形界面）
    ├── engine.py              #   检索引擎：加载两阶段模型，提供 search() 接口
    └── app.py                 #   窗口程序：输入查询 → 精排 Top50 / 召回 Top50 两个页签
```

## 运行步骤

### 1. 环境配置

建议使用 conda 创建独立环境（Python 3.10），完整命令见 `docs/环境配置命令-本机.txt`（GPU 服务器见 `docs/环境配置命令-服务器.txt`）：

```bash
conda create --name search python=3.10
conda activate search
pip install -r requirements.txt    # CPU 版本；GPU 版本请按 docs/ 中的命令安装 paddlepaddle-gpu
conda install -c conda-forge hnswlib   # Windows 无 MSVC 编译环境时用这条安装 hnswlib
```

> 本机没有 CUDA/cuDNN 运行库时使用 CPU 版 paddlepaddle 即可：单条 query 的召回 + 排序推理在 CPU 上只需数秒（`recall.py` 本身即按 CPU 设计）。首次运行会自动从百度源下载 ERNIE-1.0 与 ERNIE-3.0-medium-zh 预训练权重（约 750MB，缓存于 `~/.paddlenlp`）。

### 2. 召回模块（recall/）

1. `finetune.py`：利用 literature_search_data 数据集对 ERNIE-1.0 双塔模型进行微调；
2. `recall.py`：对测试集 query 进行语义召回（构建/加载 HNSW 索引，召回 top50）；
3. `evaluate.py`：对召回结果计算 recall@N，生成 result.tsv（记录 recall@1/5/10/20/50）。

### 3. 排序模块（rank/）

1. `train_pairwise.py`：训练 pairwise 排序模型，并在最后进行评估；
2. `predict_pairwise.py`：利用训练好的模型对测试集进行排序。

### 4. 完整检索系统（search_system/）

准备好召回模型权重（`recall_model_state.pdparams`）、排序模型权重（`rank_model_state.pdparams`）与 HNSW 索引文件（`final_index.bin`）后，运行 `search.py` 进行端到端检索。

> 说明：模型权重（*.pdparams）、索引文件（*.bin）、排序训练集（`rank/rank_dataset/train_pairwise.csv`）与召回结果文件（`recall/recall_result_file/`）单文件体积超过 GitHub 限制，未纳入版本管理；权重与索引请按上述步骤自行训练生成，或参见各 `model_param/README.md`。语料库 `corpus.csv` 已随仓库提供（`recall/recall_dataset/` 与 `search_system/` 各一份）。

### 5. 桌面客户端（desktop_app/）

图形界面版检索工具（Tkinter，无需额外安装依赖）：

```bash
cd desktop_app
python app.py
```

启动后窗口内输入查询词，回车或点击「查询」，即可查看：

- **精排结果 · Top50**：召回结果经 ERNIE-3.0 pairwise 模型重排后的最终排序（含排序分）；
- **召回结果 · Top50**：按语义相似度（召回分）排序的候选文献。

窗口底部状态栏显示检索耗时与条数；双击结果行可复制文献标题。模型只在启动时加载一次（约 20 秒），之后每次查询约 5~10 秒（CPU）。

## 效果指标

| 指标 | 数值 |
|------|------|
| 召回 recall@20 | 0.813 |
| 排序模型 AUC | 0.773 |

## 备注

本项目为独立个人学习/开发项目，代码同时提供 `.py` 脚本与 `.ipynb` notebook 两种形式，内容一致，任选其一运行即可。
