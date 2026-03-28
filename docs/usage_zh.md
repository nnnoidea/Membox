# Membox 详细使用说明（中文指南）

> 本文档面向**首次使用 Membox** 的用户，提供从环境搭建到完整评测的端到端可执行指南。

---

## 目录

1. [项目简介](#1-项目简介)
2. [目录结构说明](#2-目录结构说明)
3. [环境要求与依赖安装](#3-环境要求与依赖安装)
4. [数据集说明与准备](#4-数据集说明与准备)
5. [配置说明](#5-配置说明)
6. [快速上手（Quickstart）](#6-快速上手quickstart)
7. [分阶段运行详解](#7-分阶段运行详解)
8. [常用命令与参数说明](#8-常用命令与参数说明)
9. [输出文件说明](#9-输出文件说明)
10. [故障排查（Troubleshooting）](#10-故障排查troubleshooting)
11. [FAQ](#11-faq)

---

## 1. 项目简介

**Membox** 是一个面向 LLM Agent 的记忆架构，旨在解决现有方法将对话流碎片化为孤立文本块、依赖向量检索重建上下文时破坏叙事连贯性的问题。

Membox 采用**双层架构**：
- **Topic Loom（话题织机）**：在记忆形成阶段直接保留时序、因果和主题结构。
- **Trace Weaver（轨迹编织器）**：将跨会话的相关事件串联成事件链，支持多跳推理。

本仓库代码**既包含记忆构建（Memory Building），也包含评测（Evaluation）流程**，以 [LoCoMo](https://huggingface.co/datasets/snap-research/locomo) 数据集为基准，支持多种 QA 任务类别（多跳、时序、开放域、单跳）的端到端评测。

论文链接：[http://arxiv.org/abs/2601.03785](http://arxiv.org/abs/2601.03785)

---

## 2. 目录结构说明

```
Membox/
├── membox.py          # 核心代码：记忆构建、事件链接、检索、生成评测全流程
├── README.md          # 英文简要说明
├── docs/
│   └── usage_zh.md    # 本文档（中文详细使用说明）
└── （运行后生成的输出目录，路径由 Config.OUTPUT_BASE_DIR 指定）
    ├── final_boxes_content.jsonl
    ├── vector_store/
    ├── simple_retrieval.jsonl
    ├── simple_retrieval.csv
    ├── generation_results.jsonl
    ├── report_generation_qa.csv
    ├── generation_metrics_summary.jsonl
    ├── token_stream.jsonl
    ├── trace_build_process.jsonl
    ├── time_traces.jsonl
    ├── build_stats.jsonl
    └── trace_stats.jsonl
```

整个项目只有**一个核心脚本 `membox.py`**，通过命令行参数 `--stage` 控制运行阶段。

---

## 3. 环境要求与依赖安装

### 3.1 系统与 Python 版本要求

| 要求 | 说明 |
|------|------|
| Python | **3.10 或以上**（代码使用了 `str \| None` 联合类型语法） |
| 操作系统 | Linux / macOS / Windows（均支持） |
| GPU/CPU | 无需 GPU；所有 LLM 调用通过 OpenAI API（或兼容接口）进行，本地只需 CPU |
| 网络 | 需要能访问 OpenAI API（或自定义兼容端点） |

### 3.2 安装依赖

```bash
pip install openai scikit-learn nltk tiktoken numpy
```

各依赖用途说明：

| 包 | 用途 |
|----|------|
| `openai` | 调用 OpenAI LLM 和 Embedding API |
| `scikit-learn` | 计算向量余弦相似度（用于检索） |
| `nltk` | BLEU 分数计算（评测指标） |
| `tiktoken` | Token 计数（用于日志和上下文长度估算） |
| `numpy` | 向量运算 |

> **提示**：建议使用虚拟环境（`venv` 或 `conda`）以避免依赖冲突。
>
> ```bash
> python -m venv venv
> source venv/bin/activate  # Windows: venv\Scripts\activate
> pip install openai scikit-learn nltk tiktoken numpy
> ```

---

## 4. 数据集说明与准备

### 4.1 使用的数据集

Membox 的实验基于 **[LoCoMo 数据集](https://huggingface.co/datasets/snap-research/locomo)**——一个专为长程对话记忆评测设计的数据集，包含多个 QA 类别：

| 类别编号 | 类别名称 |
|---------|---------|
| 1 | Single-Hop（单跳） |
| 2 | Multi-Hop（多跳） |
| 3 | Temporal（时序） |
| 4 | Open Domain（开放域） |
| 5 | （跳过，代码中自动排除） |

### 4.2 数据格式

原始数据文件为 **JSON 数组**，每个元素对应一段对话，结构如下：

```json
[
  {
    "conversation": {
      "speaker_a": "Alice",
      "speaker_b": "Bob",
      "session_1": [
        {"speaker": "Alice", "text": "Hi, how are you?"},
        {"speaker": "Bob",   "text": "I'm great, thanks!"}
      ],
      "session_1_date_time": "2023-01-15",
      "session_2": [
        {"speaker": "Alice", "text": "Did you go to the concert last week?"},
        {"speaker": "Bob",   "text": "Yes, it was amazing!"}
      ],
      "session_2_date_time": "2023-01-22"
    },
    "qa": [
      {
        "id": "q1",
        "question": "Where did Bob go last week?",
        "answer": "concert",
        "category": 1,
        "evidence": [{"session": "session_2", "utterance_id": 1}]
      }
    ]
  }
]
```

关键字段说明：
- `conversation.speaker_a` / `speaker_b`：两位说话人的名字
- `conversation.session_N`：第 N 个会话的消息列表，每条消息含 `speaker`（说话人）和 `text`（内容）
- `conversation.session_N_date_time`：第 N 个会话的时间戳
- `qa`：问答列表，每项含 `question`（问题）、`answer`（标准答案）、`category`（问题类别，1-4）、`evidence`（关联的对话证据）

### 4.3 获取数据

```bash
# 方式一：通过 Hugging Face Datasets 下载 LoCoMo
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('snap-research/locomo', split='test')
import json
with open('locomo_test.json', 'w') as f:
    json.dump(list(ds), f, ensure_ascii=False, indent=2)
"
```

或者直接前往 [https://huggingface.co/datasets/snap-research/locomo](https://huggingface.co/datasets/snap-research/locomo) 手动下载。

下载后将文件路径填入 `Config.RAW_DATA_FILE`（见下节配置说明）。

---

## 5. 配置说明

打开 `membox.py`，找到顶部的 `Config` 类，修改以下**必填字段**：

```python
class Config:
    # ===== 必填 =====
    API_KEY = "your-openai-api-key"          # OpenAI API 密钥
    BASE_URL = "https://api.openai.com/v1"   # API 端点（如使用 Azure 或本地代理，修改此处）
    RAW_DATA_FILE = "path/to/locomo_test.json"   # 原始数据文件的绝对或相对路径
    OUTPUT_BASE_DIR = "path/to/output"           # 输出目录的绝对或相对路径

    # ===== 常用可选参数（按需修改）=====
    LIMIT_CONVERSATIONS = 10     # 处理前 N 条对话（调试时可设为 1 或 2 加速）
    LIMIT_SESSIONS = None        # None 表示不限制会话数量
    LLM_MODEL = "gpt-4o-mini"   # 使用的 LLM 模型名称
    EMBEDDING_MODEL = "text-embedding-3-small"  # 使用的 Embedding 模型名称
    BUILD_PREV_MSGS = 2          # 构建记忆时参考的前驱消息数
    TOP_K_RETRIEVE = 20          # 检索阶段返回的候选记忆箱数量
    ANSWER_TOP_N = 5             # 生成答案时使用的 Top-N 记忆箱数
```

> **安全提示**：请勿将含有真实 API 密钥的 `membox.py` 提交到公开仓库。建议通过环境变量传入密钥，然后在代码中读取：
>
> ```python
> import os
> API_KEY = os.environ.get("OPENAI_API_KEY", "")
> ```

---

## 6. 快速上手（Quickstart）

以下是从零开始运行完整评测流程的最小示例（假设已完成依赖安装和配置）：

```bash
# 第 1 步：克隆仓库
git clone https://github.com/nnnoidea/Membox.git
cd Membox

# 第 2 步：安装依赖
pip install openai scikit-learn nltk tiktoken numpy

# 第 3 步：修改 membox.py 中的 Config 类（填写 API_KEY、RAW_DATA_FILE、OUTPUT_BASE_DIR）

# 第 4 步：运行完整流程（build → trace → retrieve → generate）
python membox.py --stage all --run-id my_first_run
```

运行完毕后，在 `OUTPUT_BASE_DIR/my_first_run/` 目录下可找到所有输出文件，其中 `generation_metrics_summary.jsonl` 包含最终的 F1 / BLEU 评测指标。

> **调试建议**：首次运行时将 `LIMIT_CONVERSATIONS = 1` 以节省 API 调用费用，确认流程无误后再扩大规模。

---

## 7. 分阶段运行详解

完整流程分为 4 个阶段，可以一次性运行（`--stage all`），也可以单独运行每个阶段：

### Stage 1：构建记忆（Build）

将原始对话切分并聚合为带有关键词、话题摘要和显式事件的**记忆箱（Memory Box）**。

```bash
python membox.py --stage build --run-id my_run
```

输入：`Config.RAW_DATA_FILE`（原始 JSON 数据）  
输出：`final_boxes_content.jsonl`、`build_stats.jsonl`、`trace_build_process.jsonl`、`vector_store/`

**内部逻辑**：
1. 读取原始对话，逐消息判断是否需要切分（调用 LLM 判断话题连续性）
2. 对每段消息序列调用 LLM 提取关键词、话题、显式事件
3. 将结果封装为记忆箱（Memory Box）并持久化

### Stage 2：事件链接（Trace）

将各记忆箱中的事件按时序和因果关系串联成**事件链（Event Trace）**，用于多跳推理。

```bash
python membox.py --stage trace --run-id my_run
```

> **注意**：只有当 `--text-modes` 中包含 `content_trace_event` 或 `trace_event` 时，Trace 阶段才会执行。否则自动跳过。

输入：`final_boxes_content.jsonl`  
输出：`time_traces.jsonl`、`trace_stats.jsonl`

### Stage 3：检索（Retrieve）

对每个 QA 问题，基于向量相似度从记忆箱中检索最相关的候选集。

```bash
python membox.py --stage retrieve --run-id my_run
```

输入：`final_boxes_content.jsonl` + `Config.RAW_DATA_FILE`（QA 数据）  
输出：`simple_retrieval.jsonl`、`simple_retrieval.csv`

### Stage 4：生成与评测（Generate）

利用检索到的记忆箱上下文，调用 LLM 生成答案，并计算 F1 / BLEU 指标。

```bash
python membox.py --stage generate --run-id my_run --answer-topn "1,3,5" --text-modes content
```

输入：`simple_retrieval.jsonl`  
输出：`generation_results.jsonl`、`report_generation_qa.csv`、`generation_metrics_summary.jsonl`

---

## 8. 常用命令与参数说明

### 完整命令格式

```bash
python membox.py [--stage STAGE] [--run-id RUN_ID] [--build-prev-msgs N] \
                 [--answer-topn TOPN] [--text-modes MODE [MODE ...]]
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--stage` | 字符串 | `all` | 运行阶段：`build`（构建记忆）、`trace`（事件链接）、`retrieve`（检索）、`generate`（生成评测）、`all`（全流程） |
| `--run-id` | 字符串 | LLM 模型名 | 本次运行的标识符，用于区分不同实验的输出目录。输出将保存在 `OUTPUT_BASE_DIR/<run-id>/` 下 |
| `--build-prev-msgs` | 整数 | `2` | 构建记忆时，判断话题连续性所参考的前驱消息数量（最小为 1） |
| `--answer-topn` | 字符串 | `"5"` | 生成答案时使用的 Top-N 记忆箱数量，支持逗号分隔的多值（如 `"1,3,5"`，会分别评测每个值） |
| `--text-modes` | 多值字符串 | `["content_trace_event"]` | 生成阶段的文本模式（可多选）：`content`（仅内容）、`content_trace_event`（内容+事件链）、`trace_event`（仅事件链）。选择含 `trace` 的模式需要先运行 trace 阶段 |

### 常用命令示例

```bash
# 1. 仅调试构建阶段（处理 1 条对话，参考前 3 条消息）
python membox.py --stage build --run-id debug --build-prev-msgs 3

# 2. 完整流程，仅使用内容模式（跳过 Trace，速度更快）
python membox.py --stage all --run-id content_only --text-modes content

# 3. 完整流程，同时评测 Top-1/3/5 三种设置
python membox.py --stage all --run-id full_eval --answer-topn "1,3,5" --text-modes content content_trace_event

# 4. 已有 build 结果，仅重新运行 generate 阶段
python membox.py --stage generate --run-id my_run --answer-topn "5" --text-modes content
```

---

## 9. 输出文件说明

运行后，以下文件将生成在 `OUTPUT_BASE_DIR/<run-id>/` 目录下：

| 文件名 | 生成阶段 | 说明 |
|--------|----------|------|
| `final_boxes_content.jsonl` | build | 所有记忆箱数据（每行一个 JSON 对象） |
| `vector_store/` | build/retrieve | 向量缓存目录（embedding 结果，避免重复调用 API） |
| `build_stats.jsonl` | build | 构建统计信息（Token 用量、Box 数量等） |
| `trace_build_process.jsonl` | build | 话题连续性判断过程日志 |
| `time_traces.jsonl` | trace | 事件链数据（每行一条事件链记录） |
| `trace_stats.jsonl` | trace | 事件链构建统计信息 |
| `simple_retrieval.jsonl` | retrieve | 检索结果（每个 QA 对应的候选记忆箱排名） |
| `simple_retrieval.csv` | retrieve | 检索结果的 CSV 格式（便于人工查看） |
| `generation_results.jsonl` | generate | 每个 QA 的生成答案及 F1/BLEU 分数 |
| `report_generation_qa.csv` | generate | 详细的 QA 报告（CSV 格式） |
| `generation_metrics_summary.jsonl` | generate | **汇总评测指标**（不同 TopN/模式下的平均 F1/BLEU） |
| `token_stream.jsonl` | build/generate | 所有 LLM 调用的 Token 用量日志 |

---

## 10. 故障排查（Troubleshooting）

### 问题 1：`AuthenticationError` / `Incorrect API key`

**原因**：`Config.API_KEY` 未正确填写或 API 密钥无效。  
**解决**：
1. 检查 `membox.py` 中 `Config.API_KEY` 是否已填入有效的 OpenAI API 密钥
2. 确认密钥未过期，可在 [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) 验证
3. 如果使用代理端点，确认 `Config.BASE_URL` 与端点地址匹配

---

### 问题 2：`FileNotFoundError: No such file or directory` (raw data file)

**原因**：`Config.RAW_DATA_FILE` 指向的文件不存在。  
**解决**：
1. 确认数据文件已下载并放置在正确路径
2. 使用绝对路径（避免相对路径的歧义），例如 `/home/user/data/locomo_test.json`
3. 检查文件是否为合法的 JSON 数组格式（可用 `python -c "import json; json.load(open('your_file.json'))"` 验证）

---

### 问题 3：Python 版本不兼容（`SyntaxError: invalid syntax`）

**原因**：代码使用了 Python 3.10+ 的 `str | None` 联合类型语法，Python 3.9 及以下不支持。  
**解决**：升级 Python 到 3.10 或以上版本。

```bash
python --version   # 确认版本 >= 3.10
```

---

### 问题 4：`ModuleNotFoundError`

**原因**：依赖包未安装。  
**解决**：

```bash
pip install openai scikit-learn nltk tiktoken numpy
```

如果 `nltk` 在计算 BLEU 时报错（如 `LookupError: punkt`），需要下载 NLTK 数据：

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

### 问题 5：`RateLimitError` / API 调用频率超限

**原因**：短时间内发送了大量 API 请求，超过了 OpenAI 的速率限制。  
**解决**：
1. 降低 `Config.LIMIT_CONVERSATIONS` 的值，减少并发请求量
2. 考虑升级 OpenAI 账户套餐以获得更高的速率限制
3. 代码会自动重试，等待片刻后重新运行

---

### 问题 6：Trace 阶段被跳过（`Trace skipped because text_mode excludes trace events`）

**原因**：当 `--text-modes` 仅包含 `content` 时，Trace 阶段会被自动跳过（属于正常行为）。  
**解决**：如果需要运行 Trace，请在 `--text-modes` 中包含 `content_trace_event` 或 `trace_event`：

```bash
python membox.py --stage all --run-id my_run --text-modes content_trace_event
```

---

### 问题 7：生成阶段提示 `Trace file missing but trace text_mode requested`

**原因**：`--text-modes` 包含了 trace 相关模式，但 `time_traces.jsonl` 不存在（trace 阶段未运行）。  
**解决**：先运行 trace 阶段：

```bash
python membox.py --stage trace --run-id my_run
python membox.py --stage generate --run-id my_run --text-modes content_trace_event
```

---

### 问题 8：输出目录无法创建

**原因**：`Config.OUTPUT_BASE_DIR` 指向的父目录不存在。  
**解决**：手动创建输出目录：

```bash
mkdir -p /path/to/your/output
```

---

## 11. FAQ

**Q1：Membox 支持哪些 LLM 模型？**

Membox 通过 OpenAI 兼容 API 调用 LLM，默认使用 `gpt-4o-mini`（LLM）和 `text-embedding-3-small`（Embedding）。任何支持 OpenAI API 协议的模型均可使用，只需修改 `Config.LLM_MODEL`、`Config.EMBEDDING_MODEL` 和 `Config.BASE_URL`。

---

**Q2：可以使用本地部署的模型（如 Ollama、vLLM）吗？**

可以，只要该服务提供与 OpenAI API 兼容的接口。修改 `Config.BASE_URL` 为本地服务地址（如 `http://localhost:11434/v1`），并设置相应的 `Config.LLM_MODEL` 和 `Config.EMBEDDING_MODEL`。注意：本地模型的 Embedding 向量维度需与后续检索逻辑兼容。

---

**Q3：每次运行会重复调用 API 吗？**

不会完全重复。向量 Embedding 结果会缓存在 `vector_store/` 目录下，相同文本的 Embedding 不会重复计算。但 LLM 的文本生成调用（话题判断、事件提取、答案生成）每次运行都会重新发起。

---

**Q4：`--run-id` 有什么用？**

`--run-id` 是本次运行的标识符，所有输出文件会存放在 `OUTPUT_BASE_DIR/<run-id>/` 下。这样可以保存多次实验结果而不互相覆盖，便于对比不同配置下的评测指标。

---

**Q5：如何只对部分数据进行测试？**

修改 `Config.LIMIT_CONVERSATIONS`（处理前 N 条对话）和 `Config.LIMIT_SESSIONS`（每条对话最多处理 N 个会话）。例如，设置 `LIMIT_CONVERSATIONS = 1` 可以快速验证整个流程是否正常运行。

---

**Q6：`--answer-topn "1,3,5"` 是什么意思？**

`--answer-topn` 接受逗号分隔的整数列表，表示在生成答案时分别使用 Top-1、Top-3、Top-5 个记忆箱作为上下文。程序会对每个 TopN 值分别运行评测，并在 `generation_metrics_summary.jsonl` 中分别报告结果，方便消融实验分析。

---

**Q7：QA 类别 5 为什么被跳过？**

代码中在检索和生成阶段均跳过了 `category == 5` 的问题（`if qa.get("category") == 5: continue`）。这是论文作者的设计选择，类别 5 对应某类特殊问题（如图片相关），不在本方法的评测范围内。

---

**Q8：评测指标如何解读？**

`generation_metrics_summary.jsonl` 中的关键指标：
- **F1**：预测答案与标准答案在词级别的 F1 分数（越高越好，满分 100）
- **BLEU**：基于 n-gram 重叠的翻译质量指标（越高越好，满分 100）

这两个指标均在各 QA 类别（Multi-Hop / Temporal / Open Domain / Single-Hop）上分别统计。

---

*如有问题，欢迎通过 [GitHub Issues](https://github.com/nnnoidea/Membox/issues) 提交反馈。*
