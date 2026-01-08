# Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents

## Introduction

MemBox is a memory architecture designed for AI agents, inspired by the continuity and hierarchical structure of human memory. Existing approaches often fragment interaction streams into isolated text chunks for storage and then rely on embedding-based retrieval to reconstruct coherence—a process that inevitably breaks narrative and logical continuity. In contrast, MemBox places topic continuity​ at its core, employing a two-tier architecture—Topic Loom​ and Trace Weaver—to preserve temporal, causal, and thematic structures directly during memory formation.
http://arxiv.org/abs/2601.03785  

## Experimental Results

The table below presents a comparison of our method against existing state-of-the-art approaches on the LoCoMo dataset. The results demonstrate that **Membox** achieves leading performance across different task categories and LLM backbones.

| Model | Method | Multi-Hop | Temporal | Open Domain | Single Hop |
| :--- | :--- | :---: | :---: | :---: | :---: |
| | | F1 / BLEU | F1 / BLEU | F1 / BLEU | F1 / BLEU |
| **GPT-4o-mini** | LoCoMo | 25.02 / 19.75 | 18.41 / 14.77 | 12.04 / 11.16 | 40.36 / 29.05 |
| | READAGENT | 9.15 / 6.48 | 12.60 / 8.87 | 5.31 / 5.12 | 9.67 / 7.66 |
| | MEMORYBANK | 5.00 / 4.77 | 9.68 / 6.99 | 5.56 / 5.94 | 6.61 / 5.16 |
| | MEMGPT | 26.65 / 17.72 | 25.52 / 19.44 | 9.15 / 7.44 | 41.04 / 34.34 |
| | A-MEM | 27.02 / 20.09 | 45.85 / 36.67 | 12.14 / 12.00 | 44.65 / 37.06 |
| | A-MEM* | 27.08 / 20.46 | 29.14 / 24.08 | 16.60 / 13.80 | 40.70 / 32.63 |
| | Mem0 | 38.72 / **27.13** | 48.93 / 40.51 | **28.64** / **21.58** | 47.65 / 38.72 |
| | Mem0* | 36.83 / 26.50 | 34.52 / 26.38 | 22.57 / 16.54 | 46.89 / 37.63 |
| | **Membox (Ours)** | **39.88** / 26.39 | **58.03** / **45.17** | 27.96 / 20.15 | **60.09** / **47.45** |
| **GPT-4o** | LoCoMo | 28.00 / 18.47 | 9.09 / 5.78 | 16.47 / 14.80 | 61.56 / 54.19 |
| | READAGENT | 14.61 / 9.95 | 4.16 / 3.19 | 8.84 / 8.37 | 12.46 / 10.29 |
| | MEMORYBANK | 6.49 / 4.69 | 2.47 / 2.43 | 6.43 / 5.30 | 8.26 / 7.10 |
| | MEMGPT | 30.36 / 22.83 | 17.29 / 13.18 | 12.24 / 11.87 | 60.18 / 53.35 |
| | A-MEM | 32.86 / 23.76 | 39.41 / 31.23 | 17.10 / 15.84 | 48.43 / 42.97 |
| | Mem0* | 42.57 / 30.92 | 44.55 / 32.60 | 23.04 / 17.62 | 48.49 / 37.00 |
| | A-MEM* | 31.66 / 23.34 | 41.11 / 34.72 | 17.45 / 15.58 | 47.04 / 41.02 |
| | **Membox (Ours)** | **48.35** / **35.10** | **65.06** / **54.81** | **30.61** / **22.58** | **61.69** / **49.36** |

**Notes**:
1.  Methods marked with * (e.g., `Mem0*` and `A-MEM*`) represent our local implementations. For these reproduced baselines, we performed hyperparameter tuning on the retrieval scale $k \in \{5, 10, 20, 30\}$ and reported the optimal performance achieved at $k=30$.
2.  The best performance in each evaluation category is highlighted in **bold**.

## Installation and Configuration

### 1. Clone the Project

```bash
git clone https://github.com/nnnoidea/Membox.git
cd Membox
```

### 2. Install Dependencies

```bash
pip install openai scikit-learn nltk tiktoken numpy
```

### 3. Config

Set the following parameters in the `Config` class in `membox.py`:

```python
class Config:
    API_KEY = "your-openai-api-key"
    BASE_URL = "https://api.openai.com/v1"  # or other compatible endpoint
    RAW_DATA_FILE = "path/to/your/raw_data.json"
    OUTPUT_BASE_DIR = "path/to/output/directory"
```

## Usage

### Run Full Pipeline

```bash
python membox.py --stage all --run-id test_run
```

### Run by Stages

- **Build Memory**:
  ```bash
  python membox.py --stage build --run-id build_run
  ```

- **Link Traces**:
  ```bash
  python membox.py --stage trace --run-id trace_run
  ```

- **Retrieve**:
  ```bash
  python membox.py --stage retrieve --run-id retrieve_run
  ```

- **Generate Answers**:
  ```bash
  python membox.py --stage generate --run-id generate_run --answer-topn "1,3,5" --text-modes content
  ```

### Parameter Explanation

- `--stage`: Run stage (build, trace, retrieve, generate, all)
- `--run-id`: Run identifier for distinguishing output directories
- `--build-prev-msgs`: Number of previous messages to consider during building
- `--answer-topn`: Number of memory boxes to use for answer generation
- `--text-modes`: Text modes (content, content_trace_event, trace_event)

## Output Files

After running, the following files will be generated in `OUTPUT_DIR`:

- `final_boxes_content.jsonl`: Memory boxes data
- `vector_store/`: Embedding vector cache
- `simple_retrieval.jsonl/csv`: Retrieval results
- `generation_results.jsonl`: Generation results
- `report_generation_qa.csv`: QA report
- `token_stream.jsonl`: Token usage log
- `trace_build_process.jsonl`: Build trace
- `time_traces.jsonl`: Time traces
- `build_stats.jsonl`: Build statistics
- `trace_stats.jsonl`: Trace statistics
- `generation_metrics_summary.jsonl`: Generation metrics summary


## Notes

- For large datasets, run in stages to monitor progress.
- Vector embeddings are cached locally for efficiency.
- Defaults to GPT-4o-mini for LLM and text-embedding-3-small for embeddings; adjust as needed.


## Contact

For questions, please contact via GitHub Issues.
