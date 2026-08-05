# Agent Memory Leaderboard Deployment Notes

This branch packages MemBox as a minimal Add/Search HTTP service for Agent Memory
Leaderboard maintainer deployment. It does not require a hosted endpoint or any
private credential from the authors.

## Fixed Version

Use this branch or the exact commit supplied by the authors. The branch is a
lightweight reproduction interface for public-code evaluation; it is not an
author endorsement of any leaderboard result before the exact run configuration,
logs, and scores are shared back for review.

```bash
git clone https://github.com/nnnoidea/Membox.git
cd Membox
git checkout leaderboard-add-search-service
```

## Docker Deployment

Build and run:

```bash
docker build -t membox-leaderboard .
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
  -e MEMBOX_DATA_DIR=/data/membox \
  -e MEMBOX_API_KEY="${MEMBOX_API_KEY:-}" \
  -e MEMBOX_REQUIRE_OPENAI=1 \
  membox-leaderboard
```

If `MEMBOX_API_KEY` is set, call Add/Search with one of:

- `Authorization: Token <MEMBOX_API_KEY>`
- `Authorization: Bearer <MEMBOX_API_KEY>`
- `X-Api-Key: <MEMBOX_API_KEY>`

If `MEMBOX_API_KEY` is empty, authentication is disabled. This is convenient for
maintainer-run Docker deployment inside an isolated evaluation worker.

## Endpoints

- Health: `GET /health`
- Add: `POST /add`
- Search: `POST /search`

The service also exposes compatibility aliases:

- `POST /v1/memories/add`
- `POST /v1/memories/search`

## Add Contract

```json
{
  "request_id": "eval:run:dataset:conv-0:chunk-0",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "memory text"
    }
  ],
  "user_id": "eval:run:dataset:conv-0",
  "session_id": "eval:run:sample:0"
}
```

The response echoes `request_id`, `user_id`, and `session_id` after the memory is
persisted and immediately searchable:

```json
{
  "success": true,
  "request_id": "eval:run:dataset:conv-0:chunk-0",
  "user_id": "eval:run:dataset:conv-0",
  "session_id": "eval:run:sample:0"
}
```

## Search Contract

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:run:dataset:conv-0",
  "top_k": 20,
  "search_mode": "trace"
}
```

`top_k` and `search_mode` are optional. When omitted, the service uses
`MEMBOX_TOP_K` and `MEMBOX_SEARCH_MODE`; those environment variables default to
the original `membox.py` `Config.TOP_K_RETRIEVE` and a mode derived from
`Config.GEN_TEXT_MODES`.

Response:

```json
{
  "data": [
    {
      "id": "box_abc123",
      "content": "remembered dialogue text",
      "score": 0.87,
      "created_at": "2026-08-05T00:00:00Z"
    }
  ]
}
```

## Implementation Notes

The wrapper implements the online Add/Search surface expected by the leaderboard:

- Add keeps memory isolated by exact `user_id`.
- Messages are written synchronously before `HTTP 200`.
- Add request IDs are treated as idempotency keys; repeated request IDs for the
  same run are ignored after the first successful write.
- Consecutive messages in the same `session_id` are grouped into MemBox-style
  topic-continuous boxes.
- If a single session is delivered across multiple Add chunks, message coverage
  indices continue monotonically within that `user_id` and `session_id`.
- After each successful Add, the service rebuilds MemBox traces for that user's
  current memory snapshot by running the original `TraceLinker`.
- Search ranks boxes by query similarity over box content, topic, keywords, and
  extracted events.
- Search supports `box` and `trace` evidence modes. In `trace` mode, Search
  returns box content together with trace-expanded event evidence produced by the
  original trace context logic when traces are available.
- The returned `content` contains only memory evidence for the platform answer
  model; it does not generate final answers.

Scope note: the service imports the original `membox.py` implementation and uses
its `MemoryBuilder`, `TopicClusterManager`, `LLMWorker`, and `EmbeddingStore`
paths for online writing and retrieval. It also runs the original `TraceLinker`
after Add and uses the original trace-event context expansion for Search
evidence. The leaderboard wrapper does not run the platform's answer model
itself; it exposes MemBox retrieval evidence through Add/Search rather than
generating final answers inside the service.

Local token accounting is disabled in this branch. The service does not
initialize a tokenizer or estimate prompt tokens before API calls, so
OpenAI-compatible model names that are unknown to `tiktoken` can still be used.

Environment variables:

| Name | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | API key used by the maintainer-run container for MemBox LLM/embedding calls. |
| `OPENAI_BASE_URL` | `Config.BASE_URL` | OpenAI-compatible API base. |
| `MEMBOX_LLM_MODEL` | `Config.LLM_MODEL` | Model used for segmentation and memory analysis. |
| `MEMBOX_EMBEDDING_MODEL` | `Config.EMBEDDING_MODEL` | Embedding model used for ranking. |
| `MEMBOX_API_TIMEOUT_SECONDS` | `Config.API_TIMEOUT_SECONDS` | Timeout for MemBox's internal LLM/embedding calls. This is separate from the evaluator's HTTP timeout when calling Add/Search. |
| `MEMBOX_API_MAX_RETRIES` | `Config.API_MAX_RETRIES` | Retry count for MemBox's internal LLM/embedding calls. This is separate from the evaluator's Add/Search retry policy. |
| `MEMBOX_TOP_K` | `Config.TOP_K_RETRIEVE` | Default number of Search results when the request omits `top_k`. |
| `MEMBOX_SEARCH_MODE` | derived from `Config.GEN_TEXT_MODES` | Default Search evidence mode: `box` or `trace`. |
| `MEMBOX_DATA_DIR` | `.membox_data` | Run output and vector-cache directory. The in-memory Add state is intended for a single evaluation run. |
| `MEMBOX_API_KEY` | empty | Optional service authentication key. |
| `MEMBOX_REQUIRE_OPENAI` | `0` | Set to `1` for official runs to fail fast when `OPENAI_API_KEY` is missing. |

Official evaluation should provide `OPENAI_API_KEY` or an OpenAI-compatible key
through the maintainer environment. The wrapper follows the original repository's
LLM and embedding code path; running without a valid key is only useful for
checking process startup, not for a valid leaderboard result.

For local plumbing tests only, set `MEMBOX_EMBEDDING_MODEL=local-char` to replace
the embedding API with deterministic character n-gram vectors. This is useful
for checking Search ranking and response shape when the embedding endpoint is
unavailable, but it is not intended for official scoring.

Recommended official-run settings:

```bash
MEMBOX_TOP_K=100
MEMBOX_SEARCH_MODE=trace
MEMBOX_API_TIMEOUT_SECONDS=1200
MEMBOX_API_MAX_RETRIES=6
MEMBOX_REQUIRE_OPENAI=1
```

`MEMBOX_API_TIMEOUT_SECONDS` and `MEMBOX_API_MAX_RETRIES` configure MemBox's
internal calls to the LLM and embedding provider. The evaluator's Add/Search HTTP
timeouts, worker counts, and request retry policy are controlled by the
evaluation platform.

## Evaluation Assumptions

MemBox is a streaming memory system with dynamic chunking. Its memory state is
constructed from the chronological flow of messages, and box boundaries are part
of the method rather than a fixed preprocessing step.

For this reason, Add requests for the same `user_id` should be delivered in
chronological chunk order, with at most one in-flight Add request per
`user_id`. Global concurrency across different users, records, or conversations
is fine. If the evaluator sends concurrent or out-of-order Add chunks for the
same memory stream, the run may not reflect MemBox's intended usage.

## Local Smoke Test

```bash
docker build -t membox-leaderboard .
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e MEMBOX_REQUIRE_OPENAI=1 \
  membox-leaderboard
```

In another shell:

```bash
curl -s http://localhost:8080/health

curl -s http://localhost:8080/add \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id":"smoke:add:1",
    "user_id":"smoke:user:1",
    "session_id":"smoke:session:1",
    "messages":[
      {"role":"user","timestamp":1704067200000,"content":"I am preparing a workshop on agent memory."},
      {"role":"assistant","timestamp":1704067201000,"content":"Noted, the workshop is about agent memory."}
    ]
  }'

curl -s http://localhost:8080/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id":"smoke:user:1",
    "query":"What workshop is the user preparing?",
    "top_k":5
  }'
```

## Author-Side Boundary

This branch is provided for third-party reproduction from public code only.
Please do not treat the result as author-endorsed until the exact commit,
configuration, logs, and scores have been shared with the authors for review.
