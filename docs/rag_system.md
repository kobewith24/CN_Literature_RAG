# RAG 问答系统（2026 升级版）

## 1. 架构

```
浏览器 (http://127.0.0.1:8080)
   │  GET /api/ask?q=...（SSE 流式）
   ▼
Spring Boot 3.5 + LangChain4j 1.23  ──┐
   │  1) 调检索服务                     │  Ollama: qwen3:4b-instruct-2507-q4_K_M
   │  2) 组装 Prompt（带编号引用）        │  (GPU, 57 tok/s)
   │  3) 流式生成                       │
   ▼                                    ▼
Python FastAPI 检索服务 (127.0.0.1:8000)
   │  查询编码（BGE-M3, GPU）
   │  ├─ 题录索引：30 万条标题+关键词（hnswlib, 1.25GB）
   │  └─ 摘要索引：39.6 万篇 CSL 论文摘要（hnswlib, 1.65GB）
   │  双路召回 50+50 → bge-reranker-v2-m3 统一重排
   ▼
返回 Top-K passages + 引用元数据（标题/学科/摘要）
```

## 2. 组件与端口

| 组件 | 目录 | 端口 | 说明 |
|---|---|---|---|
| 检索服务 | `retrieval/` | 8000 | BGE-M3 + hnswlib + reranker（GPU） |
| 应用服务 | `rag-service/` | 8080 | Spring Boot + LangChain4j，SSE 流式问答 |
| 生成模型 | Ollama | 11434 | qwen3:4b-instruct-2507-q4_K_M（2.33GB, Q4） |

## 3. 接口

| 接口 | 说明 |
|---|---|
| `GET /api/ask?q=...&k=5` | SSE 流式问答：`sources` → `token`* → `done` |
| `GET /api/search?q=...&k=10` | 纯检索（JSON，不生成） |
| `GET /` | 内置问答页面 |
| `GET :8000/health` | 检索服务健康检查 |
| `GET :8000/retrieve?q=...&k=&mode=hybrid\|title\|doc` | 双路召回 + 重排 |

## 4. 数据

| 语料 | 规模 | 来源 |
|---|---|---|
| 题录（标题+关键词） | 300,000 | PaddleNLP literature_search_data（原项目） |
| 摘要（标题+摘要+关键词） | 395,927 | CSL 中文科学文献数据集（HF `neuclir/csl`） |

> 实测两套语料**标题重叠率约 0.6%**（300k × 395,927，严格/前缀匹配），
> 说明来源不同、互为补充：题录库负责"找文献"，摘要库负责"读内容答问题"。

## 5. 效果

- 标题级召回（同一 dev.csv，19,940 对）：recall@20 **0.8794**（旧 ERNIE 0.7698，+11.0 pt）
- 重排（1,000 对）：AUC 0.694（中位数）/ 0.755（阈值 0.7），Spearman 0.396
- 双路召回（混合，100 候选）+ 重排：**0.76 s/查询**（GPU，fp16）
- 生成速度：**57 tok/s**（RTX 4060，Q4 量化）；短答案端到端约 2 s，长答案约 14 s
- RAG 评测（60 题合成集）：检索命中 **95%**、引用命中 **95%**，详见 [rag_eval.md](rag_eval.md)

> 全部推理（编码、重排、生成）均在 GPU（cuda:0）上运行；启动时会校验设备，
> CUDA 不可用则直接报错，不会静默回退 CPU。

## 6. 启动

双击项目根目录 `启动RAG.bat`，或手动：

```bash
# 1) Ollama 与模型
ollama serve
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 2) 检索服务（首次加载索引约 40 秒）
conda activate rag
cd retrieval && python -m uvicorn service:app --host 127.0.0.1 --port 8000

# 3) 应用服务
cd rag-service
set JAVA_HOME=D:\Java\jdk-21
mvn -s maven-settings.xml package -DskipTests
java -jar target/rag-service-1.0.0.jar
```

浏览器打开 http://127.0.0.1:8080/

## 7. 已知问题与修复记录

- **Ollama 损坏**：本机 Ollama 的 `lib\ollama` 目录混入 989 个旧文件（官方仅 81 个），
  导致 `llama-server.exe` 报 `0xC000007B Bad Image`。已用官方 release 包
  （v0.40.6 `ollama-windows-amd64.zip`）全量替换该目录修复，模型与配置未受影响。
- **Maven Central 不可达**：`rag-service/maven-settings.xml` 配了阿里云镜像，用
  `mvn -s maven-settings.xml` 构建。
- **Paddle GPU**：`search` 环境通过 `sitecustomize.py` 自动加载 cuDNN 8（pip 包）与
  torch 自带的 CUDA 12 运行库，旧链路已可 GPU 复现。
