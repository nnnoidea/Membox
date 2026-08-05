"""Add/Search service wrapper for the original MemBox implementation.

This file intentionally keeps protocol handling separate from the research code:
the online service delegates topic-continuity decisions, box analysis, box
formatting, and embedding calls to the classes defined in ``membox.py``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from membox import Config, EmbeddingStore, LLMWorker, MemoryBuilder, TopicClusterManager, TraceLinker


LOG_LEVEL = os.environ.get("MEMBOX_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("membox-service")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _default_search_mode() -> str:
    modes = {str(mode).strip() for mode in (Config.GEN_TEXT_MODES or [])}
    return "trace" if modes.intersection({"content_trace_event", "trace_event"}) else "box"


class Settings:
    data_dir = Path(os.environ.get("MEMBOX_DATA_DIR", ".membox_data"))
    api_key = os.environ.get("MEMBOX_API_KEY", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    openai_base_url = os.environ.get("OPENAI_BASE_URL", Config.BASE_URL).strip()
    llm_model = os.environ.get("MEMBOX_LLM_MODEL", Config.LLM_MODEL).strip()
    embedding_model = os.environ.get("MEMBOX_EMBEDDING_MODEL", Config.EMBEDDING_MODEL).strip()
    api_timeout_seconds = float(os.environ.get("MEMBOX_API_TIMEOUT_SECONDS", str(Config.API_TIMEOUT_SECONDS)))
    api_max_retries = _env_int("MEMBOX_API_MAX_RETRIES", Config.API_MAX_RETRIES)
    default_top_k = _env_int("MEMBOX_TOP_K", Config.TOP_K_RETRIEVE)
    search_mode = os.environ.get("MEMBOX_SEARCH_MODE", _default_search_mode()).strip().lower()
    require_openai = _env_bool("MEMBOX_REQUIRE_OPENAI", False)


def configure_original_membox(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Config.API_KEY = settings.openai_api_key or "EMPTY"
    Config.BASE_URL = settings.openai_base_url
    Config.LLM_MODEL = settings.llm_model
    Config.EMBEDDING_MODEL = settings.embedding_model
    Config.API_TIMEOUT_SECONDS = settings.api_timeout_seconds
    Config.API_MAX_RETRIES = settings.api_max_retries
    Config.TOP_K_RETRIEVE = settings.default_top_k
    Config.OUTPUT_BASE_DIR = str(settings.data_dir / "runs")
    Config.apply_run_id("leaderboard_service")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_timestamp(value: Optional[Union[int, float, str]]) -> str:
    if value is None or value == "":
        return now_iso()
    try:
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, (int, float)):
            seconds = float(value) / 1000.0 if value > 10_000_000_000 else float(value)
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    return str(value)


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1)
    timestamp: Optional[Union[int, float, str]] = None


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    options: Optional[list[str]] = None
    user_id: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=1000)
    search_mode: Optional[str] = None


class SearchItem(BaseModel):
    id: str
    content: str
    score: Optional[float] = None
    created_at: Optional[str] = None


class SearchResponse(BaseModel):
    data: list[SearchItem]


class UserRuntime:
    def __init__(self, user_id: str, sample_id: int, worker: LLMWorker):
        self.user_id = user_id
        self.sample_id = sample_id
        self.builder = MemoryBuilder(worker)
        self.builder.cluster = TopicClusterManager(worker)
        self.builder.boxes = []
        self.builder.msgs = []
        self.builder.bid = 0
        self.meta = {"speaker_a": "A", "speaker_b": "B"}
        self.seen_request_ids: set[str] = set()
        self.session_message_counts: dict[str, int] = {}
        self.search_counter = 0
        self.open_revision = 0
        self.open_box_snapshot: Optional[dict[str, Any]] = None
        self.open_box_revision: Optional[int] = None
        self.trace_events_by_box: dict[Any, list[str]] = {}


class MemBoxService:
    OPEN_BOX_ID_BASE = 1_000_000_000_000
    OPEN_BOX_ID_USER_STRIDE = 1_000_000_000

    def __init__(self, settings: Settings):
        self.settings = settings
        self._validate_settings()
        configure_original_membox(settings)
        self.worker = LLMWorker()
        self.lock = threading.RLock()
        self.users: dict[str, UserRuntime] = {}
        self.next_sample_id = 0
        if settings.require_openai and not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when MEMBOX_REQUIRE_OPENAI=1")

    def _validate_settings(self) -> None:
        if self.settings.default_top_k < 1 or self.settings.default_top_k > 1000:
            raise RuntimeError("MEMBOX_TOP_K must be between 1 and 1000")
        if self.settings.search_mode not in {"box", "trace"}:
            raise RuntimeError("MEMBOX_SEARCH_MODE must be either 'box' or 'trace'")

    def _runtime(self, user_id: str) -> UserRuntime:
        if user_id not in self.users:
            self.users[user_id] = UserRuntime(user_id, self.next_sample_id, self.worker)
            self.next_sample_id += 1
        return self.users[user_id]

    def _to_original_msg(self, message: Message) -> dict[str, Any]:
        return {
            "role": message.role,
            "text": message.content.strip(),
            "time": normalize_timestamp(message.timestamp),
        }

    @staticmethod
    def _session_sort_index(session_id: str) -> int:
        digest = abs(hash(session_id))
        return digest % 1_000_000

    def add(self, request: AddRequest) -> AddResponse:
        with self.lock:
            runtime = self._runtime(request.user_id)
            if request.request_id not in runtime.seen_request_ids:
                next_idx = runtime.session_message_counts.get(request.session_id, 0) + 1
                for idx, message in enumerate(request.messages, start=next_idx):
                    runtime.builder._process(
                        self._to_original_msg(message),
                        runtime.meta,
                        sample_id=runtime.sample_id,
                        session_id=request.session_id,
                        idx=idx,
                    )
                runtime.session_message_counts[request.session_id] = next_idx + len(request.messages) - 1
                runtime.seen_request_ids.add(request.request_id)
                runtime.open_revision += 1
                self._rebuild_traces(runtime)
        return AddResponse(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        search_mode = self._search_mode(request)
        top_k = request.top_k if request.top_k is not None else self.settings.default_top_k

        with self.lock:
            runtime = self._runtime(request.user_id)
            boxes = self._current_boxes(runtime, refresh_open=False)
            if search_mode == "trace":
                trace_events_by_box = {
                    box_id: list(events)
                    for box_id, events in runtime.trace_events_by_box.items()
                }
            else:
                trace_events_by_box = {}
            runtime.search_counter += 1
            search_id = runtime.search_counter

        query_parts = [request.query]
        if request.options:
            query_parts.extend(request.options)
        query = "\n".join(query_parts)
        qvec = self.worker.get_embedding(query, note=f"service_search_{search_id}_question")

        scored: list[tuple[float, dict[str, Any]]] = []
        store = EmbeddingStore(self.worker, sample_id=runtime.sample_id)
        for box in boxes:
            score = self._score_box(query, qvec, box, store)
            scored.append((score, box))
        store.flush()
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[SearchItem] = []
        for score, box in scored[:top_k]:
            content = self._evidence_content(box, trace_events_by_box.get(box.get("box_id"), []), search_mode)
            if not content.strip():
                continue
            box_id = box.get("box_id")
            results.append(
                SearchItem(
                    id=f"box_{box_id}",
                    content=content,
                    score=float(score),
                    created_at=str(box.get("start_time") or now_iso()),
                )
            )
        return SearchResponse(data=results)

    def _search_mode(self, request: SearchRequest) -> str:
        mode = (request.search_mode or self.settings.search_mode).strip().lower()
        if mode not in {"box", "trace"}:
            raise HTTPException(status_code=400, detail={"reason": "search_mode must be 'box' or 'trace'"})
        return mode

    def _current_boxes(self, runtime: UserRuntime, refresh_open: bool) -> list[dict[str, Any]]:
        boxes = [dict(box) for box in runtime.builder.boxes]
        open_box = self._open_box(runtime, refresh=refresh_open)
        if open_box is not None:
            boxes.append(dict(open_box))
        return boxes

    def _open_box_id(self, runtime: UserRuntime) -> int:
        return (
            self.OPEN_BOX_ID_BASE
            + runtime.sample_id * self.OPEN_BOX_ID_USER_STRIDE
            + runtime.open_revision
        )

    def _open_box(self, runtime: UserRuntime, refresh: bool) -> Optional[dict[str, Any]]:
        if not runtime.builder.msgs:
            runtime.open_box_snapshot = None
            runtime.open_box_revision = None
            return None
        if (
            not refresh
            and runtime.open_box_snapshot is not None
            and runtime.open_box_revision == runtime.open_revision
        ):
            return dict(runtime.open_box_snapshot)
        box = self._materialize_open_box(runtime)
        runtime.open_box_snapshot = dict(box) if box is not None else None
        runtime.open_box_revision = runtime.open_revision if box is not None else None
        return dict(box) if box is not None else None

    def _materialize_open_box(self, runtime: UserRuntime) -> Optional[dict[str, Any]]:
        if not runtime.builder.msgs:
            return None
        builder = MemoryBuilder(self.worker)
        builder.cluster = TopicClusterManager(self.worker)
        builder.msgs = [dict(msg) for msg in runtime.builder.msgs]
        builder.bid = self._open_box_id(runtime)
        builder.boxes = []
        builder._seal(runtime.meta, sample_id=runtime.sample_id, session_id=runtime.builder.msgs[-1].get("_temp_session_id"))
        if not builder.boxes:
            return None
        return builder.boxes[-1]

    def _rebuild_traces(self, runtime: UserRuntime) -> None:
        boxes = self._current_boxes(runtime, refresh_open=True)
        runtime.trace_events_by_box = {}

        os.makedirs(os.path.dirname(Config.FINAL_CONTENT_FILE), exist_ok=True)
        with open(Config.FINAL_CONTENT_FILE, "w", encoding="utf-8") as f:
            for box in boxes:
                f.write(json.dumps(box, ensure_ascii=False) + "\n")

        for path in (Config.TIME_TRACE_FILE, Config.TRACE_PROMPT_LOG_FILE):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("", encoding="utf-8")

        if not boxes:
            return

        TraceLinker(self.worker, trace_metrics=Config.TRACE_METRICS).run()
        runtime.trace_events_by_box = self._load_trace_events_by_box(runtime, boxes)

    def _load_trace_events_by_box(self, runtime: UserRuntime, boxes: list[dict[str, Any]]) -> dict[Any, list[str]]:
        box_ids = {box.get("box_id") for box in boxes if box.get("box_id") is not None}
        trace_events_by_box: dict[Any, list[str]] = {box_id: [] for box_id in box_ids}
        seen_by_box: dict[Any, set[str]] = {box_id: set() for box_id in box_ids}
        if not os.path.exists(Config.TIME_TRACE_FILE):
            return trace_events_by_box

        with open(Config.TIME_TRACE_FILE, "r", encoding="utf-8") as f:
            traces = [json.loads(line) for line in f if line.strip()]

        for trace in traces:
            if trace.get("sample_id") != runtime.sample_id:
                continue
            if trace.get("metric") != "content_event_topic_kw":
                continue
            event_lines = TraceLinker._trace_event_lines(trace)
            for box_id in trace.get("box_ids", []) or []:
                if box_id not in box_ids:
                    continue
                for event_line in event_lines:
                    if event_line in seen_by_box[box_id]:
                        continue
                    seen_by_box[box_id].add(event_line)
                    trace_events_by_box[box_id].append(event_line)

        for box in boxes:
            box_id = box.get("box_id")
            if box_id is None:
                continue
            trace_events_by_box.setdefault(box_id, [])
        return trace_events_by_box

    def _score_box(self, query: str, qvec: list[float], box: dict[str, Any], store: EmbeddingStore) -> float:
        box_key = f"service_user_{box.get('sample_id')}_box_{box.get('box_id')}"
        features = box.get("features", {})
        content = features.get("content_text", "")
        retrieval_text = f"{content} {features.get('events_text', '')} {features.get('topic_kw_text', '')}".strip()
        vec = store.get_vector(
            box_key,
            "content_event_topic_kw",
            retrieval_text,
            note=f"service_{box_key}_content_event_topic_kw",
        )
        try:
            return float(cosine_similarity([qvec], [vec])[0][0]) if vec else -1.0
        except Exception:
            return -1.0

    @staticmethod
    def _evidence_content(box: dict[str, Any], trace_events: list[str], search_mode: str) -> str:
        features = box.get("features", {})
        content = str(features.get("content_text") or "")
        events: list[str] = []
        if search_mode == "trace":
            events = [str(e).strip() for e in trace_events if str(e).strip()]
        if search_mode == "trace" and not events:
            events = [str(e).strip() for e in (features.get("events") or []) if str(e).strip()]
        parts = [content]
        if events:
            parts.append("Events:\n" + "\n".join(events))
        return "\n\n".join(part for part in parts if part)


app = FastAPI(title="MemBox Leaderboard Add/Search Service", version="1.0.0")
settings = Settings()
service: Optional[MemBoxService] = None


def get_service() -> MemBoxService:
    global service
    if service is None:
        service = MemBoxService(settings)
    return service


async def verify_auth(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    if not settings.api_key:
        return

    provided = None
    if x_api_key:
        provided = x_api_key.strip()
    elif authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() in {"token", "bearer"}:
            provided = parts[1].strip()

    if provided != settings.api_key:
        raise HTTPException(status_code=401, detail={"reason": "invalid memory system key"})


@app.get("/health")
async def health() -> dict[str, Any]:
    ready = True
    reason = None
    try:
        get_service()
    except Exception as exc:
        ready = False
        reason = str(exc)[:500]
    return {
        "status": "ok" if ready else "error",
        "service": "membox",
        "llm_enabled": bool(settings.openai_api_key),
        "time": now_iso(),
        "detail": reason,
    }


@app.post("/add", response_model=AddResponse, dependencies=[Depends(verify_auth)])
@app.post("/v1/memories/add", response_model=AddResponse, dependencies=[Depends(verify_auth)])
async def add_memory(payload: AddRequest) -> AddResponse:
    try:
        return get_service().add(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Add failed")
        raise HTTPException(status_code=500, detail={"reason": str(exc)[:500]}) from exc


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_auth)])
@app.post("/v1/memories/search", response_model=SearchResponse, dependencies=[Depends(verify_auth)])
async def search_memory(payload: SearchRequest) -> SearchResponse:
    try:
        return get_service().search(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail={"reason": str(exc)[:500]}) from exc


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MEMBOX_HOST", "0.0.0.0")
    port = int(os.environ.get("MEMBOX_PORT", "8080"))
    uvicorn.run("membox_service:app", host=host, port=port, log_level=LOG_LEVEL.lower())
