# MemBox: Weaving Topic Continuity into Long-Range Memory for LLM Agents

## Agent Memory Leaderboard Deployment

For Agent Memory Leaderboard maintainer deployment, use the
`leaderboard-add-search-service` branch and follow
[`README_LEADERBOARD.md`](README_LEADERBOARD.md). That branch exposes MemBox as
a Dockerized synchronous Add/Search service and does not require author-hosted
endpoints or private credentials.

## Introduction

MemBox is a memory architecture for long-range conversational memory in LLM agents. It is built around topic continuity: instead of storing isolated chunks and relying only on nearest-neighbor retrieval at question time, MemBox first builds coherent memory boxes and then links them with event-level traces that preserve temporal, causal, and thematic structure.

The current version adds a trace-aware context construction step during answer generation. After retrieving the top memory boxes, MemBox uses the events associated with those boxes as trace seeds, expands them through the Trace Weaver graph, and appends the resulting event context to the answer prompt. This gives the model both the retrieved box content and the broader event chain needed for long-range questions.

Paper: http://arxiv.org/abs/2601.03785

## Method Update: Trace-Aware Context

The main change in the current code is in the generation stage:

1. Retrieve memory boxes using the box-level content, event, topic, and keyword representation.
2. Collect candidate events from the retrieved boxes.
3. If `--trace-event-topn` is an integer, rank candidate events by question-event embedding similarity and keep the top events as trace seeds.
4. Expand selected seed events to their full traces and deduplicate event lines.
5. Build the final prompt from retrieved box content plus an `Events:` block containing the expanded trace context.

`--trace-event-topn all` uses all events from the retrieved boxes as trace seeds. Smaller values such as `1` or `2` provide a more compact trace context. In the main experiments, `Membox-Compact` uses content top-k=10 with only Topic Loom box content, while `Membox-Trace` uses content top-k=10 and event top-k=2.

This differs from the previous released README/code path, where the reported MemBox results were from content-only memory boxes. The trace-aware context improves the strongest categories most clearly on temporal, open-domain, and multi-hop questions.

## Experimental Results

The table below reports the main LoCoMo results from the paper. Scores are F1 / BLEU-1. `Membox-Compact` retrieves Topic Loom boxes and provides only box content to the QA model, while `Membox-Trace` augments retrieved boxes with Trace Weaver events.

| Model | Method | Multi-Hop | Temporal | Open Domain | Single Hop |
| :--- | :--- | :---: | :---: | :---: | :---: |
| | | F1 / BLEU-1 | F1 / BLEU-1 | F1 / BLEU-1 | F1 / BLEU-1 |
| **GPT-4o-mini** | LoCoMo | 25.02 / 19.75 | 18.41 / 14.77 | 12.04 / 11.16 | 40.36 / 29.05 |
| | ReadAgent | 9.15 / 6.48 | 12.60 / 8.87 | 5.31 / 5.12 | 9.67 / 7.66 |
| | MemoryBank | 5.00 / 4.77 | 9.68 / 6.99 | 5.56 / 5.94 | 6.61 / 5.16 |
| | MEMGPT | 26.65 / 17.72 | 25.52 / 19.44 | 9.15 / 7.44 | 41.04 / 34.34 |
| | A-MEM | 27.02 / 20.09 | 45.85 / 36.67 | 12.14 / 12.00 | 44.65 / 37.06 |
| | A-MEM* | 27.08 / 20.46 | 29.14 / 24.08 | 16.60 / 13.80 | 40.70 / 32.63 |
| | Mem0 | 38.72 / 27.13 | 48.93 / 40.51 | 28.64 / 21.58 | 47.65 / 38.72 |
| | Mem0* | 36.83 / 26.50 | 34.52 / 26.38 | 22.57 / 16.54 | 46.89 / 37.63 |
| | **Membox-Compact** | 39.88 / 26.39 | 58.03 / 45.17 | 27.96 / 20.15 | 60.09 / 47.45 |
| | **Membox-Trace** | **41.19 / 27.49** | **59.63 / 46.52** | **30.36 / 22.52** | **61.18 / 48.99** |
| **GPT-4o** | LoCoMo | 28.00 / 18.47 | 9.09 / 5.78 | 16.47 / 14.80 | 61.56 / **54.19** |
| | ReadAgent | 14.61 / 9.95 | 4.16 / 3.19 | 8.84 / 8.37 | 12.46 / 10.29 |
| | MemoryBank | 6.49 / 4.69 | 2.47 / 2.43 | 6.43 / 5.30 | 8.26 / 7.10 |
| | MEMGPT | 30.36 / 22.83 | 17.29 / 13.18 | 12.24 / 11.87 | 60.18 / 53.35 |
| | A-MEM | 32.86 / 23.76 | 39.41 / 31.23 | 17.10 / 15.84 | 48.43 / 42.97 |
| | Mem0* | 42.57 / 30.92 | 44.55 / 32.60 | 23.04 / 17.62 | 48.49 / 37.00 |
| | A-MEM* | 31.66 / 23.34 | 41.11 / 34.72 | 17.45 / 15.58 | 47.04 / 41.02 |
| | **Membox-Compact** | 48.35 / 35.10 | 65.06 / **54.81** | 30.61 / 22.58 | 61.69 / 49.36 |
| | **Membox-Trace** | **50.48 / 38.17** | **66.61** / 54.15 | **38.77 / 28.19** | **62.56** / 48.95 |

Notes:

1. Methods marked with `*` are local reproductions.
2. For re-implemented baselines, retrieval depth is tuned over k in {5, 10, 20, 30}, and the optimal result is reported.
3. The best result for each model, category, and metric is highlighted in bold.

## Trace Context Ablation

The table below isolates the effect of adding trace context to the MemBox generation prompt under the fixed main settings. `Membox-Compact` uses content top-k=10. `Membox-Trace` uses content top-k=10 and event top-k=2.

| Model | Setting | Multi-Hop | Temporal | Open Domain | Single Hop |
| :--- | :--- | :---: | :---: | :---: | :---: |
| | | F1 / BLEU-1 | F1 / BLEU-1 | F1 / BLEU-1 | F1 / BLEU-1 |
| GPT-4o-mini | Membox-Compact | 39.88 / 26.39 | 58.03 / 45.17 | 27.96 / 20.15 | 60.09 / 47.45 |
| GPT-4o-mini | Membox-Trace | 41.19 / 27.49 | 59.63 / 46.52 | 30.36 / 22.52 | 61.18 / 48.99 |
| GPT-4o | Membox-Compact | 48.35 / 35.10 | 65.06 / 54.81 | 30.61 / 22.58 | 61.69 / 49.36 |
| GPT-4o | Membox-Trace | 50.48 / 38.17 | 66.61 / 54.15 | 38.77 / 28.19 | 62.56 / 48.95 |

## Installation and Configuration

```bash
git clone https://github.com/nnnoidea/Membox.git
cd Membox
pip install openai scikit-learn nltk tiktoken numpy
```

Set the following fields in `Config`:

```python
class Config:
    API_KEY = "your-openai-api-key"
    BASE_URL = "https://api.openai.com/v1"
    RAW_DATA_FILE = "path/to/locomo10.json"
    OUTPUT_BASE_DIR = "path/to/output/directory"
```

## Usage

Run the full pipeline:

```bash
python membox.py --stage all --run-id test_run
```

Run by stage:

```bash
python membox.py --stage build --run-id test_run
python membox.py --stage trace --run-id test_run
python membox.py --stage retrieve --run-id test_run
python membox.py --stage generate --run-id test_run --answer-topn 10 --text-modes content_trace_event --trace-event-topn 2
```

Trace context:

```bash
python membox.py --stage generate --run-id test_run --answer-topn 10 --text-modes content_trace_event --trace-event-topn 2
```

Content-only baseline:

```bash
python membox.py --stage generate --run-id test_run --answer-topn 10 --text-modes content
```
