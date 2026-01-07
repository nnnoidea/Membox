# Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents

## Project Introduction

Membox is a dialogue memory construction and retrieval system based on Large Language Models (LLMs). The system analyzes dialogue data, segments it into meaningful "memory boxes," and supports content-based retrieval and answer generation.

## Main Features

- **Memory Construction**: Segments dialogues into topic-related memory boxes, extracting keywords, topics, and events.
- **Trace Linking**: Links related events into temporal sequence traces based on similarity and LLM judgment.
- **Simple Retrieval**: Content retrieval based on vector similarity.
- **Answer Generation**: Generates accurate answers using retrieved memories and evaluates performance (F1, BLEU, etc.).


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

## Evaluation Metrics

- **F1 Score**: F1 match between predicted answer and ground truth
- **BLEU Score**: Similarity evaluation based on BLEU
- **Token Usage**: LLM call statistics and token consumption

## Notes

- For large datasets, run in stages to monitor progress.
- Vector embeddings are cached locally for efficiency.
- Defaults to GPT-4o-mini for LLM and text-embedding-3-small for embeddings; adjust as needed.


## Contact

For questions, please contact via GitHub Issues.
