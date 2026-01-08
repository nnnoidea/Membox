import argparse
import json
import os
import logging
import csv
import re
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


# ================= 1. 全局配置 =================
class Config:
    API_KEY = ""
    BASE_URL = ""

    RAW_DATA_FILE = ""
    OUTPUT_BASE_DIR = ""
    RUN_ID: str | None = None

    # 路径会在 apply_run_id 时按 run_id 重写
    OUTPUT_DIR = OUTPUT_BASE_DIR
    FINAL_CONTENT_FILE = os.path.join(OUTPUT_DIR, "final_boxes_content.jsonl")
    VECTOR_DIR = os.path.join(OUTPUT_DIR, "vector_store")
    SIMPLE_RETRIEVAL_JSONL = os.path.join(OUTPUT_DIR, "simple_retrieval.jsonl")
    SIMPLE_RETRIEVAL_CSV = os.path.join(OUTPUT_DIR, "simple_retrieval.csv")
    GENERATION_RESULT_FILE = os.path.join(OUTPUT_DIR, "generation_results.jsonl")
    GENERATION_REPORT_CSV = os.path.join(OUTPUT_DIR, "report_generation_qa.csv")
    TOKEN_LOG_FILE = os.path.join(OUTPUT_DIR, "token_stream.jsonl")
    BUILD_TRACE_FILE = os.path.join(OUTPUT_DIR, "trace_build_process.jsonl")
    TIME_TRACE_FILE = os.path.join(OUTPUT_DIR, "time_traces.jsonl")
    TRACE_PROMPT_LOG_FILE = os.path.join(OUTPUT_DIR, "trace_prompts.jsonl")
    TRACE_STATS_FILE = os.path.join(OUTPUT_DIR, "trace_stats.jsonl")

    LIMIT_CONVERSATIONS = 10
    LIMIT_SESSIONS = None  # None 表示不限制
    TOP_K_RETRIEVE = 20
    TOP_K_GENERATE = 5
    BUILD_PREV_MSGS = 2
    CHECKPOINT_EVERY_SAMPLE = True
    TRACE_SIMILARITY_THRESHOLD = 0.5

    TRACE_METRICS = ["content_event_topic_kw"]

    # 生成阶段使用的 box 数量（答案上下文 TopN）
    ANSWER_TOP_N = 5
    # 默认仅运行 content 文本模式，按需开启 trace 模式
    GEN_TEXT_MODES = ["content_trace_event"]

    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    # --- 提示词 ---
    PROMPT_MSG_CONTINUATION = """Please determine whether the current message continues with the main topic of the previous messages. Only answer Yes/No/Partially Shifted.  
previous messages: {ref}
current message: {curr}
Answer:"""

    PROMPT_DIALOG_EXTRACT = """Generate a structured analysis of the provided dialog by performing the following tasks:

1. **Identifying salient keywords:** Extract 3-8 most salient nouns, named entities, and key terminology that represent core concepts. Avoid common words (e.g., "good", "see") and prioritize specificity.

2. **Determining the core topic:** In one clear phrase, state the primary subject or objective of the discussion based on the actual content.

3. **Extracting explicit event and plan mentions:** Identify and list only the **events, factual developments, or specific future plans** that are **explicitly mentioned** in the dialog. Follow these strict rules:
    3.1. **Focus on Verbatim or Near-Verbatim Content:** Each extracted item must be directly grounded in the dialog text. Do not infer, summarize, or combine information to create new "events."
    3.2. **Distinguish Event Types:**
        - **Past/Completed Events:** Actions or occurrences that are stated as having happened (e.g., "I went to...", "We completed the project").
        - **Established Facts/Changes:** Concrete facts or changes presented as already true (e.g., "I am now the team lead", "The system is down").
        - **Explicit Future Plans:** Specific plans for the future mentioned by the speakers (e.g., "We will meet on Friday", "I'm planning to visit Paris").
    3.3. **Exclude Non-Events:** Do NOT include:
        - General states of being (e.g., "I'm swamped", "I'm happy").
        - Questions, greetings, or expressions of intent without a plan (e.g., "We should talk sometime").
        - Vague aspirations or possibilities.
    3.4. **Framing:** Phrase each extracted item as a concise, standalone clause that captures the core of what was mentioned.

**Output Format:** Provide the analysis as a valid JSON object with the following exact keys:
{{
"keywords": ["keyword1", "keyword2", ...],
"topic": "clear topic phrase",
"explicit_mentions": [
    "A mentioned past event or established fact",
    "A mentioned specific future plan",
    // ... more as needed
]
}}
Content to analyze: {text}"""


    

    PROMPT_TRACE_EVENT_FILTER = """You are a narrative coherence analyzer for constructing and maintaining event memory chains. Your task is to filter events from a new event list (Event List B) that are directly related to an existing event chain (Event Chain A).

**Core Task:**
Event Chain A represents an existing sequence of events (could be one or multiple events). Event List B is a set of newly observed events. Analyze each event in B to determine whether it should:
1. Serve as a **direct continuation** of Event Chain A (directly related to A's core narrative)
2. Be considered **unrelated** to Event Chain A (independent or belonging to a different event stream)

**Analysis Principles:**
- Identify the **core theme/activity** from Event Chain A's overall narrative
- Assess narrative continuity: Does the event from B advance, develop, or resolve A's core activity?
- Consider temporal/causal logic: Does the event naturally follow A's chain in time or logic?

**Decision Criteria:**
An event from B is **related** to Event Chain A if it:
1. Continues the **same core activity** as A's chain (not just similar topic)
2. Provides **progress, outcome, solution, or direct consequence** to A's chain
3. Is a **logical/temporal successor** to A's chain

An event from B is **unrelated** to Event Chain A if it:
1. Initiates a **new, distinct activity** (even if topic is similar)
2. Is a **parallel but independent** event to A's core activity
3. Concerns a **different aspect** unrelated to A's main thread
4. Is a **generic response** without specific progression

**Output Format:**
Strictly use this JSON format:
{{
    "chain_summary": "Brief summary of Event Chain A's core theme (1-2 sentences)",
    "related_events": ["Exact text of related events from B"],
    "unrelated_events": ["Exact text of unrelated events from B"],
    "reasoning": {{
        "related_reasons": ["Brief explanation for each related event"],
        "unrelated_reasons": ["Brief explanation for each unrelated event"]
    }}
}}

**Example 1:**
Event Chain A: ["I'm planning a weekend hike", "I checked the weather forecast", "I bought hiking shoes"]
Event List B: ["I mapped out the hiking route", "I replied to work emails", "I contacted hiking partners", "Went to see a movie in the evening"]

Output:
{{
    "chain_summary": "Preparations for a weekend hiking trip",
    "related_events": ["I mapped out the hiking route", "I contacted hiking partners"],
    "unrelated_events": ["I replied to work emails", "Went to see a movie in the evening"],
    "reasoning": {{
        "related_reasons": [
            "Mapping the route is a concrete step in hike preparation",
            "Contacting partners directly advances the hiking activity"
        ],
        "unrelated_reasons": [
            "Work emails concern a different domain (work vs. recreation)",
            "Movie watching is a separate leisure activity"
        ]
    }}
}}

**Example 2:**
Event Chain A: ["The project encountered technical difficulties", "The team met to discuss solutions"]
Event List B: ["I researched relevant documentation", "Decided to adopt a new framework", "Had pizza for lunch", "Client sent new requirements"]

Output:
{{
    "chain_summary": "Addressing technical challenges in a project",
    "related_events": ["I researched relevant documentation", "Decided to adopt a new framework"],
    "unrelated_events": ["Had pizza for lunch", "Client sent new requirements"],
    "reasoning": {{
        "related_reasons": [
            "Researching documentation directly addresses the technical problem",
            "Deciding on a new framework represents a solution to the technical challenge"
        ],
        "unrelated_reasons": [
            "Lunch is a routine activity unrelated to problem-solving",
            "New client requirements initiate a separate work thread"
        ]
    }}
}}

**Now analyze:**
Event Chain A: {content_a} (Note: This is an existing event chain)
Event List B: {content_b} (Note: This is a new event list)
Output your analysis.
"""

    PROMPT_TRACE_INIT = """You are an event chain constructor for building coherent memory structures. Your task is to analyze a set of events and organize them into logical chains.

**Task:**
Given a set of events, identify the primary narrative thread and any associated events that form a coherent event chain.

**Process:**
1. Analyze all events to identify the most prominent theme or activity
2. Connect events that share temporal, causal, or thematic relationships
3. Form the most coherent sequence possible
4. Identify any events that don't fit into the main narrative thread

**Output Format:**
{{
    "primary_chain": ["Events forming the most coherent narrative, in logical order"],
    "secondary_chains": [["Other potential chains, if any"]],
    "isolated_events": ["Events that don't fit into any chain"],
    "chain_summary": "Brief description of the primary chain's theme and context"
}}

**Examples:**

Example 1:
Events: ["I woke up at 7 AM", "I checked my email", "I had breakfast", "Then I went for a run"]

Output:
{{
    "primary_chain": ["I woke up at 7 AM", "I had breakfast", "Then I went for a run"],
    "secondary_chains": [],
    "isolated_events": ["I checked my email"],
    "chain_summary": "Morning routine including waking, eating, and exercise"
}}

Example 2:
Events: ["Started a new project at work", "Researched design patterns", "Met with the client", "Created initial wireframes", "Had lunch with a colleague"]

Output:
{{
    "primary_chain": ["Started a new project at work", "Researched design patterns", "Created initial wireframes"],
    "secondary_chains": [["Met with the client"]],
    "isolated_events": ["Had lunch with a colleague"],
    "chain_summary": "Work project initiation and initial design phase"
}}

**Now analyze:**
Events: {events}
Output your analysis in JSON format.
"""

    PROMPT_QA_ANSWER = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:

You have access to memories from two speakers in conversations. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:

1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
5. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory timestamp. Ignore the reference while answering the question.
6. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
7. The answer should be less than 5-6 words.

# APPROACH (Think step by step):

1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories:

{memories}

Question: {question}

Answer:"""

    @classmethod
    def sanitize_run_id(cls, run_id: str | None) -> str:
        rid = (run_id or cls.LLM_MODEL or "default").strip()
        rid = re.sub(r"[^A-Za-z0-9_.-]+", "_", rid)
        return rid or "default"

    @classmethod
    def apply_run_id(cls, run_id: str | None):
        rid = cls.sanitize_run_id(run_id)
        cls.RUN_ID = rid
        cls.OUTPUT_DIR = os.path.join(cls.OUTPUT_BASE_DIR, rid)
        cls.FINAL_CONTENT_FILE = os.path.join(cls.OUTPUT_DIR, "final_boxes_content.jsonl")
        cls.VECTOR_DIR = os.path.join(cls.OUTPUT_DIR, "vector_store")
        cls.SIMPLE_RETRIEVAL_JSONL = os.path.join(cls.OUTPUT_DIR, "simple_retrieval.jsonl")
        cls.SIMPLE_RETRIEVAL_CSV = os.path.join(cls.OUTPUT_DIR, "simple_retrieval.csv")
        cls.GENERATION_RESULT_FILE = os.path.join(cls.OUTPUT_DIR, "generation_results.jsonl")
        cls.GENERATION_REPORT_CSV = os.path.join(cls.OUTPUT_DIR, "report_generation_qa.csv")
        cls.TOKEN_LOG_FILE = os.path.join(cls.OUTPUT_DIR, "token_stream.jsonl")
        cls.BUILD_TRACE_FILE = os.path.join(cls.OUTPUT_DIR, "trace_build_process.jsonl")
        cls.TIME_TRACE_FILE = os.path.join(cls.OUTPUT_DIR, "time_traces.jsonl")
        cls.TRACE_PROMPT_LOG_FILE = os.path.join(cls.OUTPUT_DIR, "trace_prompts.jsonl")
        cls.BUILD_STATS_FILE = os.path.join(cls.OUTPUT_DIR, "build_stats.jsonl")
        cls.GEN_SUMMARY_FILE = os.path.join(cls.OUTPUT_DIR, "generation_metrics_summary.jsonl")
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.VECTOR_DIR, exist_ok=True)

    @classmethod
    def vector_file(cls, sample_id: int) -> str:
        return os.path.join(cls.VECTOR_DIR, f"sample_{sample_id}.json")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ================= 2. 基础服务类 =================
class TraceLogger:
    @staticmethod
    def log(file_path, data):
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


class TokenAnalyzer:
    stage_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})

    @staticmethod
    def log_usage(usage, note, extra=None):
        if not usage:
            return
        stage = None
        if extra and "stage" in extra:
            stage = extra["stage"]
        entry = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "note": note,
            "in": usage.prompt_tokens,
            "out": getattr(usage, "completion_tokens", 0),
        }
        if extra:
            entry.update(extra)
        with open(Config.TOKEN_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if stage:
            stats = TokenAnalyzer.stage_stats[stage]
            stats["calls"] += 1
            stats["prompt"] += usage.prompt_tokens
            stats["completion"] += getattr(usage, "completion_tokens", 0)
            stats["total"] += getattr(usage, "total_tokens", usage.prompt_tokens + getattr(usage, "completion_tokens", 0))

    @staticmethod
    def get_stage_stats(stage: str) -> Dict[str, float]:
        return TokenAnalyzer.stage_stats.get(stage, {"calls": 0, "prompt": 0, "completion": 0, "total": 0})


def evidence_to_targets(evidence_list, boxes):
    """Map evidence like 'D1:3' to box_ids covering that session/message."""
    targets = set()
    if not evidence_list:
        return []

    session_map = {}
    for b in boxes:
        cov = b.get("coverage", {})
        raw_sid = cov.get("session_id")
        sid_norm = None
        try:
            if isinstance(raw_sid, str) and raw_sid.startswith("session_"):
                sid_norm = int(raw_sid.split("_")[1])
            else:
                sid_norm = int(raw_sid)
        except Exception:
            sid_norm = raw_sid
        session_map.setdefault(sid_norm, []).append(b)

    for ev in evidence_list:
        try:
            part = ev.split(":")
            sid = int(part[0][1:])
            msg_idx = int(part[1])
            if sid not in session_map:
                continue
            for b in session_map[sid]:
                cov = b.get("coverage", {})
                if cov.get("start_idx", 0) <= msg_idx <= cov.get("end_idx", 0):
                    targets.add(b["box_id"])
        except Exception:
            continue

    return sorted(list(targets))


class EmbeddingStore:
    """Lazy embedding cache: per-sample load, fetch-or-compute, flush on demand."""

    def __init__(self, worker: "LLMWorker", sample_id: int):
        self.worker = worker
        self.sample_id = sample_id
        self.path = Config.vector_file(sample_id)
        self.data: Dict[str, Dict[str, Any]] = {}
        self.dirty = False
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get_vector(self, key: str, field: str, text: str, note: str, stage: str | None = None) -> List[float]:
        if not text:
            return []
        if key in self.data and field in self.data[key]:
            return self.data[key][field]
        vec = self.worker.get_embedding(text, note=note)
        self.data.setdefault(key, {})[field] = vec
        self.dirty = True
        return vec

    def ensure_key(self, key: str):
        if key not in self.data:
            self.data[key] = {}
            self.dirty = True

    def flush(self):
        if not self.dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f)
        self.dirty = False


class LLMWorker:
    def __init__(self):
        self.client = OpenAI(base_url=Config.BASE_URL, api_key=Config.API_KEY)
        self.encoding = tiktoken.encoding_for_model(Config.LLM_MODEL)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    def get_embedding(self, text, note="Emb"):
        try:
            if not text:
                return [0.0] * 1536
            resp = self.client.embeddings.create(
                input=text.replace("\n", " "), model=Config.EMBEDDING_MODEL
            )
            emb = None
            try:
                emb = resp.data[0].embedding
            except Exception:
                emb = None
            return emb if emb is not None else [0.0] * 1536
        except Exception:
            return [0.0] * 1536

    def chat_completion(self, prompt, note="Completion", json_mode=False, extra=None):
        try:
            kwargs = {
                "model": Config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            extra_payload = {"prompt_tokens_est": self.count_tokens(prompt)}
            if extra:
                extra_payload.update(extra)
            TokenAnalyzer.log_usage(resp.usage, note, extra_payload)
            return resp.choices[0].message.content.strip()
        except Exception:
            return "{}" if json_mode else ""

    def check_relation(self, text_prev_list, text_curr, note="Relation"):
        ref_text = "\n".join(text_prev_list)
        prompt = Config.PROMPT_MSG_CONTINUATION.format(ref=ref_text, curr=text_curr)
        res = self.chat_completion(prompt, note=note, extra={"prompt_tokens_est": self.count_tokens(prompt), "stage": "build"})
        return "Yes" if "yes" in res.lower() else "No"


# ================= 3. 构建 (Build) =================
class TopicClusterManager:
    def __init__(self, worker: LLMWorker):
        self.worker = worker

    def process_new_box(self, new_box, sample_id):
        prefix = f"S{sample_id}_B{new_box['box_id']}"
        content_str = self._get_content_str(new_box)

        ps1_prompt = Config.PROMPT_DIALOG_EXTRACT.format(text=content_str)
        ps1_raw = self.worker.chat_completion(
            ps1_prompt,
            note=f"{prefix}_NotePS1",
            json_mode=True,
            extra={"prompt_tokens_est": self.worker.count_tokens(ps1_prompt), "stage": "build"},
        )
        topic = ""
        keywords_txt = ""
        events = []
        events_text = ""
        try:
            d = json.loads(ps1_raw)
            topic = str(d.get("topic", "") or "").strip()
            kws = d.get("keywords", [])
            if isinstance(kws, list):
                keywords_txt = ", ".join([str(k).strip() for k in kws if str(k).strip()])
            else:
                keywords_txt = str(kws).strip()

            evs = d.get("explicit_mentions", [])
            if isinstance(evs, list):
                events = [str(ev).strip() for ev in evs if str(ev).strip()]
                events_text = " | ".join(events)
            else:
                events = []
                events_text = str(evs).strip()
        except:
            print("something wrong")
        topic_kw_text = f"{topic} {keywords_txt}".strip()

        new_box["runtime_info"] = {
            "topic": topic,
            "topic_kw_text": topic_kw_text,
            "keywords": keywords_txt,
            "events": events,
            "events_text": events_text,
        }

        TraceLogger.log(
            Config.BUILD_TRACE_FILE,
            {
                "type": "box_created",
                "sample_id": sample_id,
                "box_id": new_box["box_id"],
                "content_preview": content_str[:100],
                "extracted": {"topic": topic, "keywords": keywords_txt, "events": events},
            },
        )

    def _get_content_str(self, box):
        header = "Start: " + str(box.get("background_info", {}).get("start_time", "Unknown"))
        lines = []
        for m in box.get("content", []):
            ts = m.get("time", "")
            lines.append(f"{ts} {m.get('role')}: {m.get('text')}")
        return "\n".join([header] + lines)


class MemoryBuilder:
    def __init__(self, worker):
        self.worker = worker
        self.cluster = None
        self.boxes = []
        self.msgs = []
        self.bid = 0
        self.token_ratios: List[float] = []
        self.msg_counts: List[int] = []
        self.box_token_pairs: List[Tuple[int, int]] = []
        self.total_boxes: int = 0

    def build_all(self):
        all_samples_boxes = []
        with open(Config.RAW_DATA_FILE, "r") as f:
            raw_list = json.load(f)[: Config.LIMIT_CONVERSATIONS]
        logger.info(f"🏗️  [BUILD] Processing {len(raw_list)} Conversations...")

        for sample_id, conversation_data in enumerate(raw_list):
            logger.info(f"   Building Sample {sample_id}...")
            self.cluster = TopicClusterManager(self.worker)
            self.boxes = []
            self.msgs = []
            self.bid = 0
            conv = conversation_data.get("conversation", {})
            static_meta = {
                "speaker_a": conv.get("speaker_a", "A"),
                "speaker_b": conv.get("speaker_b", "B"),
            }
            keys = sorted(
                [k for k in conv.keys() if k.startswith("session_") and len(k) < 12],
                key=lambda x: int(x.split("_")[1]),
            )

            for k in keys[: Config.LIMIT_SESSIONS]:
                t_key = f"{k}_date_time"
                s_time = conv.get(t_key, "Unknown")
                if "json" in str(s_time):
                    s_time = "Unknown"

                if k not in conv or not isinstance(conv[k], list):
                    continue
                for idx, m in enumerate(conv[k], start=1):
                    if not isinstance(m, dict):
                        continue
                    msg = {"role": m.get("speaker"), "text": m.get("text", "").strip(), "time": s_time}
                    current_session_id = k
                    self._process(msg, static_meta, sample_id, current_session_id, idx)
            self._seal(static_meta, sample_id, current_session_id if "current_session_id" in locals() else None)
            all_samples_boxes.extend(self.boxes)
            if Config.CHECKPOINT_EVERY_SAMPLE:
                self.save_incremental(self.boxes, append=True)
        return all_samples_boxes

    def _process(self, msg, meta, sample_id, session_id, idx=None):
        msg["_temp_session_id"] = session_id
        if idx is not None:
            msg["_temp_idx"] = idx

        if not self.msgs:
            self.msgs.append(msg)
            return

        last_session = self.msgs[-1].get("_temp_session_id")
        if last_session != session_id:
            self._seal(meta, sample_id, last_session)
            self.msgs.append(msg)
            return

        if len(self.msgs) == 1:
            self.msgs.append(msg)
            return

        prev_msgs = []
        if self.msgs:
            window = min(len(self.msgs), max(1, Config.BUILD_PREV_MSGS))
            prev_slice = self.msgs[-window:]
            prev_msgs = [f"{m['role']}: {m['text']}" for m in prev_slice]

        curr_msg_str = f"{msg['role']}: {msg['text']}"
        res = self.worker.check_relation(prev_msgs, curr_msg_str, note=f"S{sample_id}_Overhead_Split")

        TraceLogger.log(
            Config.BUILD_TRACE_FILE,
            {
                "type": "split_check",
                "sample_id": sample_id,
                "session_id": session_id,
                "prev_msgs": prev_msgs,
                "curr_msg": curr_msg_str,
                "decision": res,
            },
        )

        if res == "Yes":
            self.msgs.append(msg)
        else:
            self._seal(meta, sample_id, last_session)
            self.msgs.append(msg)

    def _seal(self, meta, sample_id, session_id):
        if not self.msgs:
            return
        start_idx = self.msgs[0].get("_temp_idx", 1)
        end_idx = self.msgs[-1].get("_temp_idx", 1)
        content_to_save = []
        for m in self.msgs:
            m_copy = m.copy()
            m_copy.pop("_temp_session_id", None)
            m_copy.pop("_temp_idx", None)
            content_to_save.append(m_copy)
        raw_box = {
            "sample_id": sample_id,
            "box_id": self.bid,
            "background_info": {"start_time": self.msgs[0]["time"]},
            "content": content_to_save,
            "coverage": {"session_id": session_id, "start_idx": start_idx, "end_idx": end_idx},
        }
        self.cluster.process_new_box(raw_box, sample_id)

        content_text = self.cluster._get_content_str(raw_box)
        rt = raw_box.get("runtime_info", {})
        messages_count = len(content_to_save)
        content_tokens = self.worker.count_tokens(content_text)
        enrich_text = " ".join([content_text, rt.get("topic", ""), rt.get("keywords", ""), rt.get("events_text", "")]).strip()
        enriched_tokens = self.worker.count_tokens(enrich_text)
        ratio = enriched_tokens / max(1, content_tokens)
        self.token_ratios.append(ratio)
        self.msg_counts.append(messages_count)
        self.box_token_pairs.append((content_tokens, enriched_tokens))

        final_box = {
            "sample_id": sample_id,
            "box_id": self.bid,
            "start_time": raw_box.get("background_info", {}).get("start_time"),
            "coverage": raw_box.get("coverage", {}),
            "features": {
                "content_text": content_text,
                "topic_kw_text": rt.get("topic_kw_text", ""),
                "events": rt.get("events", []),
                "events_text": rt.get("events_text", ""),
            },
        }

        self.bid += 1
        self.boxes.append(final_box)
        self.total_boxes += 1
        self.msgs = []

    def _write_boxes(self, boxes):
        mode = "a" if os.path.exists(Config.FINAL_CONTENT_FILE) else "w"
        os.makedirs(os.path.dirname(Config.FINAL_CONTENT_FILE), exist_ok=True)
        with open(Config.FINAL_CONTENT_FILE, mode, encoding="utf-8") as f:
            for b in boxes:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")

    def save(self, boxes):
        self._write_boxes(boxes)
        logger.info(f"✅ [BUILD] Complete. Saved {len(boxes)} boxes (appended).")

    def save_incremental(self, boxes, append=True):
        if not boxes:
            return
        self._write_boxes(boxes)
        logger.info(f"✅ [BUILD] Checkpoint saved: +{len(boxes)} boxes (appended)")

    def summarize_and_log(self):
        total_boxes = self.total_boxes or len(self.token_ratios)
        avg_msg = sum(self.msg_counts) / total_boxes if total_boxes else 0
        avg_ratio = sum(self.token_ratios) / total_boxes if total_boxes else 0
        total_content_tokens = sum(p[0] for p in self.box_token_pairs)
        total_enriched_tokens = sum(p[1] for p in self.box_token_pairs)
        llm_stats = TokenAnalyzer.get_stage_stats("build")
        total_messages = sum(self.msg_counts)
        boxes_denom = max(total_boxes, 1)
        msgs_denom = max(total_messages, 1)
        llm_calls = llm_stats.get("calls", 0)
        summary = {
            "run_id": Config.RUN_ID,
            "timestamp": datetime.now().isoformat(),
            "boxes": total_boxes,
            "total_messages": total_messages,
            "avg_messages_per_box": round(avg_msg, 3),
            "avg_token_ratio_enriched_over_content": round(avg_ratio, 3),
            "total_content_tokens": total_content_tokens,
            "total_enriched_tokens": total_enriched_tokens,
            "llm_calls_build": llm_stats.get("calls", 0),
            "llm_prompt_tokens": llm_stats.get("prompt", 0),
            "llm_completion_tokens": llm_stats.get("completion", 0),
            "llm_total_tokens": llm_stats.get("total", 0),
            "avg_llm_calls_per_box": round(llm_calls / boxes_denom, 3),
            "avg_llm_calls_per_message": round(llm_calls / msgs_denom, 3),
            "avg_prompt_tokens_per_box": round(llm_stats.get("prompt", 0) / boxes_denom, 3),
            "avg_completion_tokens_per_box": round(llm_stats.get("completion", 0) / boxes_denom, 3),
            "avg_total_tokens_per_box": round(llm_stats.get("total", 0) / boxes_denom, 3),
            "avg_prompt_tokens_per_message": round(llm_stats.get("prompt", 0) / msgs_denom, 3),
            "avg_completion_tokens_per_message": round(llm_stats.get("completion", 0) / msgs_denom, 3),
            "avg_total_tokens_per_message": round(llm_stats.get("total", 0) / msgs_denom, 3),
        }
        os.makedirs(os.path.dirname(Config.BUILD_STATS_FILE), exist_ok=True)
        with open(Config.BUILD_STATS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        logger.info(
            "ℹ️ Build stats | boxes=%s msgs=%s avg_msg=%.2f avg_ratio=%.3f llm_calls=%s (per_box=%.3f per_msg=%.3f) tokens(prompt/out/total)=(%s/%s/%s) per_box=(%.3f/%.3f/%.3f) per_msg=(%.3f/%.3f/%.3f)",
            total_boxes,
            total_messages,
            avg_msg,
            avg_ratio,
            llm_calls,
            llm_calls / boxes_denom,
            llm_calls / msgs_denom,
            llm_stats.get("prompt", 0),
            llm_stats.get("completion", 0),
            llm_stats.get("total", 0),
            llm_stats.get("prompt", 0) / boxes_denom,
            llm_stats.get("completion", 0) / boxes_denom,
            llm_stats.get("total", 0) / boxes_denom,
            llm_stats.get("prompt", 0) / msgs_denom,
            llm_stats.get("completion", 0) / msgs_denom,
            llm_stats.get("total", 0) / msgs_denom,
        )


class TraceLinker:
    """离线将 box 按 topic+keywords 近邻 + LLM 判定合并为 trace。"""

    def __init__(self, worker: LLMWorker, trace_metrics: List[str] | None = None):
        self.worker = worker
        self.trace_metrics = trace_metrics or Config.TRACE_METRICS
        self.relation_cache: Dict[Tuple[int, int, int], bool] = {}

    @staticmethod
    def _box_text(box: Dict[str, Any]) -> str:
        feat = box.get("features", {})
        return feat.get("content_text", "")

    def _entry(self, box: Dict[str, Any], order: int, events: List[str] | None = None) -> Dict[str, Any]:
        events_list = events if events is not None else box.get("features", {}).get("events", [])
        events_clean = [str(e).strip() for e in (events_list or []) if str(e).strip()]
        return {
            "box_id": box["box_id"],
            "start_time": str(box.get("start_time", "Unknown")),
            "events": events_clean,
            "order": order,
        }

    @staticmethod
    def _metric_text(box: Dict[str, Any], metric: str) -> str:
        feat = box.get("features", {})
        content = feat.get("content_text", "")
        evt = feat.get("events_text", "")
        topic_kw = feat.get("topic_kw_text", "")
        if metric == "content_event_topic_kw":
            return f"{content} {evt} {topic_kw}".strip()
        return content

    @staticmethod
    def _trace_event_lines(trace: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for entry in trace.get("entries", []):
            ts = str(entry.get("start_time", "Unknown"))
            for ev in entry.get("events") or []:
                ev_clean = str(ev).strip()
                if ev_clean:
                    lines.append(f"{ts}: {ev_clean}")
        return lines

    def _llm_event_filter(self, trace: Dict[str, Any], events: List[str], sample_id: int) -> Tuple[set, set]:
        chain_text = "\n".join(self._trace_event_lines(trace)) or "None"
        events_text = "\n".join(events) or "None"
        prompt = Config.PROMPT_TRACE_EVENT_FILTER.format(content_a=chain_text, content_b=events_text)
        res = self.worker.chat_completion(
            prompt,
            note=f"S{sample_id}_TraceLinker_EventFilter",
            extra={"prompt_tokens_est": self.worker.count_tokens(prompt), "stage": "trace"},
        )
        TraceLogger.log(Config.TRACE_PROMPT_LOG_FILE, {
            "type": "event_filter",
            "sample_id": sample_id,
            "trace_id": trace.get("trace_id"),
            "prompt": prompt,
            "response": res,
        })
        related = set()
        unrelated = set()
        try:
            parsed = json.loads(res)
            related_list = parsed.get("related_events") or []
            unrelated_list = parsed.get("unrelated_events") or []
            for ev in events:
                if ev in related_list:
                    related.add(ev)
                elif ev in unrelated_list:
                    unrelated.add(ev)
        except Exception:
            related = set(events)
        if not related and not unrelated:
            related = set(events)
        return related, unrelated

    def _llm_init_chain(self, events: List[str], sample_id: int) -> Dict[str, Any]:
        prompt = Config.PROMPT_TRACE_INIT.format(events="\n".join(events))
        res = self.worker.chat_completion(
            prompt,
            note=f"S{sample_id}_TraceLinker_Init",
            extra={"prompt_tokens_est": self.worker.count_tokens(prompt), "stage": "trace"},
        )
        TraceLogger.log(Config.TRACE_PROMPT_LOG_FILE, {
            "type": "init_chain",
            "sample_id": sample_id,
            "prompt": prompt,
            "response": res,
        })
        try:
            return json.loads(res)
        except Exception:
            return {}


    def run(self):
        if not os.path.exists(Config.FINAL_CONTENT_FILE):
            logger.error("❌ Need build outputs first.")
            return

        with open(Config.FINAL_CONTENT_FILE, "r") as f:
            boxes = [json.loads(line) for line in f]

        metrics = [m for m in (self.trace_metrics or []) if m == "content_event_topic_kw"] or ["content_event_topic_kw"]

        limit = Config.LIMIT_CONVERSATIONS
        total_boxes = 0
        for sample_id in sorted({b["sample_id"] for b in boxes}):
            if limit is not None and sample_id >= limit:
                continue

            store = EmbeddingStore(self.worker, sample_id)
            sample_boxes = [b for b in boxes if b["sample_id"] == sample_id]
            sample_boxes.sort(key=lambda x: x["box_id"])
            total_boxes += len(sample_boxes)

            sample_traces: List[Dict[str, Any]] = []

            for metric in metrics:
                traces_merged: List[Dict[str, Any]] = []

                for box in sample_boxes:
                    events_raw = box.get("features", {}).get("events", []) or []
                    events = [str(e).strip() for e in events_raw if str(e).strip()]

                    if not events:
                        continue

                    trace_lookup = {t["trace_id"]: t for t in traces_merged}
                    selected_trace_ids = set()

                    # Stage 1: Select candidate traces based on similarity
                    for ev_idx, ev in enumerate(events):
                        ev_vec = store.get_vector(
                            f"S{sample_id}_B{box['box_id']}_E{ev_idx}",
                            "event",
                            ev,
                            note=f"S{sample_id}_B{box['box_id']}_event",
                        )
                        if not ev_vec:
                            continue

                        best_trace_id = None
                        best_score = -1.0
                        for tr in traces_merged:
                            trace_best = -1.0
                            for entry_idx, entry in enumerate(tr.get("entries", [])):
                                for tev_idx, tev in enumerate(entry.get("events") or []):
                                    key = f"S{sample_id}_T{tr['trace_id']}_E{entry_idx}_{tev_idx}"
                                    tvec = store.get_vector(
                                        key,
                                        "event",
                                        tev,
                                        note=f"S{sample_id}_T{tr['trace_id']}_event",
                                    )
                                    if not tvec:
                                        continue
                                    try:
                                        score = cosine_similarity([ev_vec], [tvec])[0][0]
                                    except Exception:
                                        score = 0.0
                                    if score > trace_best:
                                        trace_best = score
                            if trace_best > best_score:
                                best_score = trace_best
                                best_trace_id = tr["trace_id"]

                        if best_trace_id is not None and best_score >= Config.TRACE_SIMILARITY_THRESHOLD:
                            selected_trace_ids.add(best_trace_id)

                    # Stage 2: LLM Filter for selected traces
                    matched_events = set()
                    for tr_id in selected_trace_ids:
                        trace = trace_lookup.get(tr_id)
                        if not trace:
                            continue
                        
                        # Pass ALL events to the LLM filter for this trace
                        related, _ = self._llm_event_filter(trace, events, sample_id)
                        if related:
                            if box["box_id"] not in trace["box_ids"]:
                                trace["box_ids"].append(box["box_id"])
                            trace["entries"].append(self._entry(box, len(trace["entries"]), list(related)))
                            matched_events.update(related)

                    # Identify unmatched events
                    unmatched_events = [e for e in events if e not in matched_events]
                    unmatched_events = [e for e in unmatched_events if e]

                    if unmatched_events:
                        def _create_trace_with_events(ev_list: List[str]):
                            ev_clean = [str(e).strip() for e in ev_list if str(e).strip()]
                            if not ev_clean:
                                return
                            entry = self._entry(box, 0, ev_clean)
                            trace_local = {
                                "sample_id": sample_id,
                                "metric": metric,
                                "trace_id": len(traces_merged),
                                "box_ids": [box["box_id"]],
                                "entries": [entry],
                            }
                            traces_merged.append(trace_local)

                        if len(unmatched_events) == 1:
                            _create_trace_with_events(unmatched_events)
                        else:
                            init_res = self._llm_init_chain(unmatched_events, sample_id) or {}
                            primary_chain = init_res.get("primary_chain") or []
                            secondary_chains = init_res.get("secondary_chains") or []
                            isolated_events = init_res.get("isolated_events") or []

                            chains_to_create: List[List[str]] = []
                            if primary_chain:
                                chains_to_create.append(primary_chain)
                            for ch in secondary_chains:
                                if ch:
                                    chains_to_create.append(ch)
                            if not chains_to_create and isolated_events:
                                chains_to_create.append(isolated_events)

                            for chain_events in chains_to_create:
                                _create_trace_with_events(chain_events)

                            if chains_to_create and isolated_events:
                                used_events = {e for chain in chains_to_create for e in chain}
                                remaining_iso = [e for e in isolated_events if e not in used_events]
                                for iso in remaining_iso:
                                    _create_trace_with_events([iso])

                for t in traces_merged:
                    texts = []
                    for entry in t["entries"]:
                        if entry["events"]:
                            texts.append(f"{entry['start_time']}: {entry['events'][0]}")
                            for e in entry["events"][1:]:
                                texts.append(e)
                    t["entries_text"] = " ".join(texts)

                sample_traces.extend(traces_merged)

            store.flush()

            if sample_traces:
                with open(Config.TIME_TRACE_FILE, "a", encoding="utf-8") as f_out:
                    for t in sample_traces:
                        f_out.write(json.dumps(t, ensure_ascii=False) + "\n")
                logger.info("✅ Trace saved for sample %s (%d traces)", sample_id, len(sample_traces))

        logger.info(f"✅ Trace linking completed. Output -> {Config.TIME_TRACE_FILE}")

        llm_stats = TokenAnalyzer.get_stage_stats("trace")
        avg_total_tokens_per_box = llm_stats.get("total", 0) / max(total_boxes, 1)
        logger.info(
            "ℹ️ Trace LLM stats | calls=%s prompt=%s completion=%s total=%s avg_total_tokens_per_box=%.3f",
            llm_stats.get("calls", 0),
            llm_stats.get("prompt", 0),
            llm_stats.get("completion", 0),
            llm_stats.get("total", 0),
            avg_total_tokens_per_box,
        )

        summary = {
            "run_id": Config.RUN_ID,
            "timestamp": datetime.now().isoformat(),
            "total_boxes": total_boxes,
            "llm_calls": llm_stats.get("calls", 0),
            "llm_prompt_tokens": llm_stats.get("prompt", 0),
            "llm_completion_tokens": llm_stats.get("completion", 0),
            "llm_total_tokens": llm_stats.get("total", 0),
            "avg_total_tokens_per_box": round(avg_total_tokens_per_box, 3),
        }
        os.makedirs(os.path.dirname(Config.TRACE_STATS_FILE), exist_ok=True)
        with open(Config.TRACE_STATS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")



class SimpleRetriever:
    """基于 content 向量与问题相似度的简易检索，不生成答案。"""

    def __init__(self, worker: LLMWorker, top_k: int = None):
        self.worker = worker
        # top_k=None means keep full ranking; otherwise cap at provided value
        self.top_k = Config.TOP_K_RETRIEVE if top_k is None else top_k
        self.all_boxes: List[Dict[str, Any]] = []
        self.trace_map: Dict[int, Dict[int, List[int]]] = {}

    def load(self):
        with open(Config.FINAL_CONTENT_FILE, "r") as f:
            self.all_boxes = [json.loads(l) for l in f]
        self.trace_map = self._load_traces(Config.TIME_TRACE_FILE)

    @staticmethod
    def _tokens(text: str) -> set:
        stop = {
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "on",
            "for",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "with",
            "by",
            "at",
            "from",
            "that",
            "this",
            "it",
            "as",
            "but",
            "if",
            "about",
            "into",
            "than",
            "then",
            "so",
            "such",
            "not",
            "no",
            "do",
            "does",
            "did",
        }
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        return {t for t in tokens if t and t not in stop}

    def _load_traces(self, path: str) -> Dict[int, Dict[int, List[int]]]:
        """Return mapping: sample_id -> box_id -> trace box_ids from file."""
        if not os.path.exists(path):
            return {}
        trace_map: Dict[int, Dict[int, List[int]]] = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    t = json.loads(line)
                    sid = t.get("sample_id")
                    ids = t.get("box_ids", []) or []
                    if sid is None:
                        continue
                    trace_map.setdefault(sid, {})
                    for bid in ids:
                        trace_map[sid][bid] = ids
        except Exception:
            return {}
        return trace_map

    def _score_and_rank(self, sample_id: int, qa: Dict[str, Any]):
        pool = [b for b in self.all_boxes if b["sample_id"] == sample_id]
        q = qa.get("question", "")
        q_id = qa.get("id", qa.get("question", ""))
        store = EmbeddingStore(self.worker, sample_id)
        qvec = store.get_vector(f"qa_{sample_id}_{q_id}", "question", q, note=f"S{sample_id}_QA_Content")
        sim_map = {}
        for b in pool:
            key = f"{sample_id}_{b['box_id']}"
            text = b.get("features", {}).get("content_text", "")
            text = f"{text} {b.get('features', {}).get('events_text', '')} {b.get('features', {}).get('topic_kw_text', '')}".strip()
            v = store.get_vector(key, "content_event_topic_kw", text, note=f"S{sample_id}_B{b['box_id']}_content_event_topic_kw")
            try:
                s = cosine_similarity([qvec], [v])[0][0] if v else -1.0
            except Exception:
                s = -1.0
            sim_map[b["box_id"]] = s
        ranked = [bid for bid, _ in sorted(sim_map.items(), key=lambda x: x[1], reverse=True)]
        # Keep full ordering for downstream reuse; top-k slice is optional
        rankings = {
            "content_event_topic_kw": ranked,
        }
        store.flush()
        target_boxes = evidence_to_targets(qa.get("evidence"), pool)
        return rankings, sim_map, target_boxes

    def run(self, result_jsonl: str, result_csv: str):
        if not os.path.exists(Config.RAW_DATA_FILE):
            logger.error("❌ No raw data file.")
            return
        self.load()

        logger.info("ℹ️ Retrieval will append results to: %s, %s", result_jsonl, result_csv)
        header_written = os.path.exists(result_csv)
        csv_file = open(result_csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        if not header_written:
            writer.writerow([
                "Sample_ID",
                "QA_ID",
                "Question",
                "Category",
                "Ranking_ContentEventTopicKW",
                "Targets",
            ])

        with open(Config.RAW_DATA_FILE, "r") as f:
            raw_list = json.load(f)[: Config.LIMIT_CONVERSATIONS]

        for sample_id, data in enumerate(raw_list):
            for qa_idx, qa in enumerate(data.get("qa", [])):
                if qa.get("category") == 5:
                    continue
                rankings, sim_maps, target_boxes = self._score_and_rank(sample_id, qa)

                writer.writerow(
                    [
                        sample_id,
                        qa_idx,
                        qa.get("question", ""),
                        qa.get("category", ""),
                        rankings.get("content_event_topic_kw", []),
                        target_boxes,
                    ]
                )

                res_entry = {
                    "sample_id": sample_id,
                    "qa_idx": qa_idx,
                    "question": qa.get("question", ""),
                    "category": qa.get("category", ""),
                    "rankings": rankings,
                    "target_boxes": target_boxes,
                }
                TraceLogger.log(result_jsonl, res_entry)
            logger.info(f"✅ Simple retrieval done for sample {sample_id}")
        csv_file.close()
        logger.info("✅ Retrieval results appended to %s", result_jsonl)



class AnswerGenerator:
    """统一的生成与评估模块，可选问题重排。"""

    def __init__(
        self,
        worker: LLMWorker,
        answer_topn: int | List[int] | None = None,
        text_modes: List[str] | None = None,
        stage_label: str = "gen",
    ):
        self.worker = worker
        if isinstance(answer_topn, list):
            self.answer_topn_list = answer_topn
        else:
            self.answer_topn_list = [answer_topn or Config.ANSWER_TOP_N or Config.TOP_K_RETRIEVE]
        self.text_modes = text_modes or Config.GEN_TEXT_MODES
        self.trace_metrics = Config.TRACE_METRICS
        self.encoding = tiktoken.encoding_for_model(Config.LLM_MODEL)
        self.box_map: Dict[int, Dict[int, str]] = {}
        self.qa_map: Dict[int, List[Dict[str, Any]]] = {}
        self.boxes_by_sample: Dict[int, List[Dict[str, Any]]] = {}
        self.trace_map: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
        self.content_totals: Dict[int, int] = defaultdict(int)
        self.aggregate: Dict[Tuple[str, str, str, str, int], Dict[str, float]] = defaultdict(lambda: {"f1_sum": 0.0, "bleu_sum": 0.0, "ctx_tokens_sum": 0.0, "count": 0})
        self.aggregate_by_category: Dict[Tuple[str, str, str, str, int, str], Dict[str, float]] = defaultdict(lambda: {"f1_sum": 0.0, "bleu_sum": 0.0, "ctx_tokens_sum": 0.0, "count": 0})
        self.conv_ctx_total: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "count": 0.0})
        self.conv_ctx_by_mode: Dict[Tuple[str, int, str], Dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "count": 0.0})
        self.stage_label = stage_label

    @staticmethod
    def _tokens(text: str) -> List[str]:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(text or "").lower())
        return [t for t in cleaned.split() if t]

    @classmethod
    def _f1(cls, pred: str, gold: Any) -> float:
        pred_tokens = cls._tokens(pred)
        gold_list = gold if isinstance(gold, list) else [gold]
        best = 0.0
        for g in gold_list:
            gold_tokens = cls._tokens(g)
            if not gold_tokens or not pred_tokens:
                overlap = 0
            else:
                overlap = 0
                gold_counts = {}
                for t in gold_tokens:
                    gold_counts[t] = gold_counts.get(t, 0) + 1
                for t in pred_tokens:
                    if t in gold_counts and gold_counts[t] > 0:
                        overlap += 1
                        gold_counts[t] -= 1
            if overlap == 0:
                f1 = 0.0
            else:
                precision = overlap / len(pred_tokens)
                recall = overlap / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall)
            best = max(best, f1)
        return best

    @classmethod
    def _bleu(cls, pred: str, gold: Any) -> float:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        pred = str(pred)
        refs = [str(g) for g in gold] if isinstance(gold, list) else [str(gold)]
        pred_tokens = nltk.word_tokenize(pred.lower())
        refs_tokens = [nltk.word_tokenize(r.lower()) for r in refs]
        smooth = SmoothingFunction().method1
        try:
            return sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        except Exception:
            return 0.0

    @staticmethod
    def _question_words(question: str) -> set:
        return set(re.findall(r"[A-Za-z0-9]+", (question or "").lower()))

    def _load_boxes(self):
        if not os.path.exists(Config.FINAL_CONTENT_FILE):
            return
        with open(Config.FINAL_CONTENT_FILE, "r") as f:
            for line in f:
                b = json.loads(line)
                sid = b.get("sample_id")
                bid = b.get("box_id")
                if sid is None or bid is None:
                    continue
                text = b.get("features", {}).get("content_text", "")
                self.box_map.setdefault(sid, {})[bid] = text
                self.boxes_by_sample.setdefault(sid, []).append({"box_id": bid, "coverage": b.get("coverage", {})})
                self.content_totals[sid] += len(self.encoding.encode(text))

    def _load_qa(self):
        if not os.path.exists(Config.RAW_DATA_FILE):
            return
        with open(Config.RAW_DATA_FILE, "r") as f:
            raw_list = json.load(f)[: Config.LIMIT_CONVERSATIONS]
        for sid, data in enumerate(raw_list):
            self.qa_map[sid] = data.get("qa", [])

    def _load_traces(self):
        traces: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        if os.path.exists(Config.TIME_TRACE_FILE):
            with open(Config.TIME_TRACE_FILE, "r") as f:
                for line in f:
                    t = json.loads(line)
                    sid = t.get("sample_id")
                    metric = t.get("metric")
                    if sid is None or metric is None:
                        continue
                    traces[sid][metric].append(t)
        self.trace_map = traces

    def _trace_events_for_box(self, sid: int, bid: int, metric: str) -> List[str]:
        traces = self.trace_map.get(sid, {}).get(metric, [])
        for tr in traces:
            if bid not in tr.get("box_ids", []):
                continue
            events_texts: List[str] = []
            for entry in tr.get("entries", []):
                evs = entry.get("events") or []
                if not evs:
                    continue
                ts = str(entry.get("start_time", "Unknown"))
                for ev in evs:
                    ev_clean = str(ev).strip()
                    if ev_clean:
                        events_texts.append(f"{ts}: {ev_clean}")
            return events_texts
        return []

    def _build_trace_contexts(self, sid: int, top_ids: List[int], trace_metric: str, mode: str) -> List[str]:
        contexts: List[str] = []
        seen_events = set()
        for bid in top_ids:
            events = self._trace_events_for_box(sid, bid, trace_metric)
            if not events:
                continue
            events_text = "\n".join(events)
            if events_text in seen_events:
                continue
            seen_events.add(events_text)
            if mode == "content_trace_event":
                content = self.box_map.get(sid, {}).get(bid)
                if not content:
                    continue
                contexts.append(f"{content}\nEvents:\n{events_text}")
            elif mode == "trace_event":
                contexts.append(f"Events:\n{events_text}")
        return contexts

    def _log_token_counts(self, context_text: str, question: str) -> Dict[str, Any]:
        return {
            "memories_tokens": len(self.encoding.encode(context_text)),
            "question_tokens": len(self.encoding.encode(question)),
            "prompt_tokens_est": len(self.encoding.encode(Config.PROMPT_QA_ANSWER.format(memories=context_text, question=question))),
        }

    def _record_metrics(self, ranking_strategy: str, metric: str, trace_metric: str | None, mode: str, topn: int, f1: float, bleu: float, ctx_tokens: int, sid: int, category: Any):
        key = (ranking_strategy, metric, trace_metric or "", mode, topn)
        agg = self.aggregate[key]
        agg["f1_sum"] += f1
        agg["bleu_sum"] += bleu
        agg["ctx_tokens_sum"] += ctx_tokens
        agg["count"] += 1

        cat_label = "uncategorized" if category is None else str(category)
        cat_key = (ranking_strategy, metric, trace_metric or "", mode, topn, cat_label)
        cat_agg = self.aggregate_by_category[cat_key]
        cat_agg["f1_sum"] += f1
        cat_agg["bleu_sum"] += bleu
        cat_agg["ctx_tokens_sum"] += ctx_tokens
        cat_agg["count"] += 1

        conv_key = (ranking_strategy, sid)
        conv_stat_total = self.conv_ctx_total[conv_key]
        conv_stat_total["tokens"] += ctx_tokens
        conv_stat_total["count"] += 1

        conv_mode_key = (ranking_strategy, sid, mode)
        conv_stat_mode = self.conv_ctx_by_mode[conv_mode_key]
        conv_stat_mode["tokens"] += ctx_tokens
        conv_stat_mode["count"] += 1

    def _write_summary(self):
        if not self.aggregate:
            return
        records = []
        for (ranking_strategy, metric, trace_metric, mode, topn), v in self.aggregate.items():
            count = max(v.get("count", 0), 1)
            records.append(
                {
                    "run_id": Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "metric": metric,
                    "trace_metric": trace_metric,
                    "text_mode": mode,
                    "topn": topn,
                    "avg_f1": round(v.get("f1_sum", 0) / count, 4),
                    "avg_bleu": round(v.get("bleu_sum", 0) / count, 4),
                    "avg_context_tokens": round(v.get("ctx_tokens_sum", 0) / count, 2),
                    "count": v.get("count", 0),
                }
            )

        for (ranking_strategy, metric, trace_metric, mode, topn, category), v in self.aggregate_by_category.items():
            count = max(v.get("count", 0), 1)
            records.append(
                {
                    "run_id": Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "metric": metric,
                    "trace_metric": trace_metric,
                    "text_mode": mode,
                    "topn": topn,
                    "category": category,
                    "avg_f1": round(v.get("f1_sum", 0) / count, 4),
                    "avg_bleu": round(v.get("bleu_sum", 0) / count, 4),
                    "avg_context_tokens": round(v.get("ctx_tokens_sum", 0) / count, 2),
                    "count": v.get("count", 0),
                    "type": "category_metrics",
                }
            )

        for (ranking_strategy, sid), stat in self.conv_ctx_total.items():
            count = max(stat.get("count", 0), 1)
            avg_ctx = stat.get("tokens", 0) / count
            content_total = self.content_totals.get(sid, 1)
            records.append(
                {
                    "run_id": Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "sample_id": sid,
                    "avg_context_tokens": round(avg_ctx, 2),
                    "content_tokens_total": content_total,
                    "avg_context_ratio_over_content": round(avg_ctx / max(content_total, 1), 4),
                    "count": stat.get("count", 0),
                    "type": "conversation_context_usage",
                }
            )

        for (ranking_strategy, sid, mode), stat in self.conv_ctx_by_mode.items():
            content_total = self.content_totals.get(sid, 1)
            count = max(stat.get("count", 0), 1)
            avg_ctx = stat.get("tokens", 0) / count
            records.append(
                {
                    "run_id": Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "sample_id": sid,
                    "text_mode": mode,
                    "avg_context_tokens": round(avg_ctx, 2),
                    "content_tokens_total": content_total,
                    "avg_context_ratio_over_content": round(avg_ctx / max(content_total, 1), 4),
                    "count": stat.get("count", 0),
                    "type": "conversation_context_usage_by_mode",
                }
            )

        os.makedirs(os.path.dirname(Config.GEN_SUMMARY_FILE), exist_ok=True)
        with open(Config.GEN_SUMMARY_FILE, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _generate_for_ranking(
        self,
        *,
        ranking_strategy: str,
        metric: str,
        trace_metric: str | None,
        mode: str,
        top_ids: List[int],
        sid: int,
        qid: int,
        question: str,
        gold: Any,
        targets: List[int],
        category: Any,
        writer: csv.writer,
        out_jsonl: str,
        topn: int,
    ):
        if mode == "content":
            contexts = [self.box_map.get(sid, {}).get(bid) for bid in top_ids if self.box_map.get(sid, {}).get(bid)]
        else:
            contexts = self._build_trace_contexts(sid, top_ids, trace_metric or "content_event_topic_kw", mode)
        if not contexts:
            return

        context_text = "\n\n".join(contexts)
        user_prompt = Config.PROMPT_QA_ANSWER.format(memories=context_text, question=question)
        note = f"S{sid}_QA_{qid}_{ranking_strategy}_{metric}_top{topn}_{trace_metric or 'content'}_{mode}"
        token_info = self._log_token_counts(context_text, question)
        ans = self.worker.chat_completion(
            user_prompt,
            note=note,
            extra={**token_info, "stage": f"{self.stage_label}:{ranking_strategy}"},
        )
        f1 = self._f1(ans, gold)
        bleu = self._bleu(ans, gold)
        ctx_tokens = token_info.get("memories_tokens", 0)

        writer.writerow(
            [
                sid,
                qid,
                ranking_strategy,
                question,
                gold,
                ans,
                f"{f1:.4f}",
                f"{bleu:.4f}",
                metric,
                trace_metric or "",
                mode,
                topn,
                top_ids,
                targets,
                category,
                ctx_tokens,
            ]
        )

        TraceLogger.log(
            out_jsonl,
            {
                "sample_id": sid,
                "qa_idx": qid,
                "ranking_strategy": ranking_strategy,
                "question": question,
                "gold": gold,
                "pred": ans,
                "f1": f1,
                "bleu": bleu,
                "metric": metric,
                "trace_metric": trace_metric,
                "text_mode": mode,
                "topn": topn,
                "topk": top_ids,
                "target_boxes": targets,
                "category": category,
                "context_tokens": ctx_tokens,
            },
        )
        self._record_metrics(ranking_strategy, metric, trace_metric, mode, topn, f1, bleu, ctx_tokens, sid, category)

    def run(self, retrieval_jsonl: str, base_out_jsonl: str, base_out_csv: str):
        if not os.path.exists(retrieval_jsonl):
            logger.error("❌ Retrieval result not found: %s", retrieval_jsonl)
            return

        self._load_boxes()
        self._load_qa()
        self._load_traces()

        logger.info("ℹ️ Generation text_modes=%s answer_topn=%s", self.text_modes, self.answer_topn_list)

        csv_base_exists = os.path.exists(base_out_csv)
        csv_base_file = open(base_out_csv, "a", newline="", encoding="utf-8")
        base_writer = csv.writer(csv_base_file)
        if not csv_base_exists:
            base_writer.writerow([
                "Sample_ID",
                "QA_ID",
                "Ranking_Strategy",
                "Question",
                "Gold",
                "Pred",
                "F1",
                "BLEU",
                "Metric",
                "Trace_Metric",
                "Text_Mode",
                "TopN",
                "TopIDs",
                "Targets",
                "Category",
                "Context_Tokens",
            ])

        with open(retrieval_jsonl, "r") as f:
            entries = [json.loads(line) for line in f]

        for ent in entries:
            sid = ent.get("sample_id")
            qid = ent.get("qa_idx")
            if sid is None or qid is None:
                continue
            qa_list = self.qa_map.get(sid, [])
            if qid >= len(qa_list):
                continue
            qa = qa_list[qid]
            question = qa.get("question", "")
            gold = qa.get("answer", "")
            category = qa.get("category")
            if category == 5:
                continue
            targets = evidence_to_targets(qa.get("evidence"), self.boxes_by_sample.get(sid, []))

            rankings = ent.get("rankings", {}) or {}
            base_rank = rankings.get("content_event_topic_kw", []) or []
            if not base_rank:
                continue

            ranking_sets = [("baseline", base_rank, base_writer, base_out_jsonl)]

            for ranking_strategy, ranking_list, writer_obj, out_path in ranking_sets:
                if not writer_obj or not out_path:
                    continue

                for topn in self.answer_topn_list:
                    top_ids = ranking_list[: topn]
                    if not top_ids:
                        continue

                    if "content" in self.text_modes:
                        self._generate_for_ranking(
                            ranking_strategy=ranking_strategy,
                            metric="content_event_topic_kw",
                            trace_metric=None,
                            mode="content",
                            top_ids=top_ids,
                            sid=sid,
                            qid=qid,
                            question=question,
                            gold=gold,
                            targets=targets,
                            category=category,
                            writer=writer_obj,
                            out_jsonl=out_path,
                            topn=topn,
                        )

                    if "content_trace_event" in self.text_modes or "trace_event" in self.text_modes:
                        for trace_metric in self.trace_metrics:
                            for mode in [m for m in self.text_modes if m in ("content_trace_event", "trace_event")]:
                                self._generate_for_ranking(
                                    ranking_strategy=ranking_strategy,
                                    metric="content_event_topic_kw",
                                    trace_metric=trace_metric,
                                    mode=mode,
                                    top_ids=top_ids,
                                    sid=sid,
                                    qid=qid,
                                    question=question,
                                    gold=gold,
                                    targets=targets,
                                    category=category,
                                    writer=writer_obj,
                                    out_jsonl=out_path,
                                    topn=topn,
                                )

        csv_base_file.close()
        self._write_summary()
        logger.info("✅ Generation complete")
def _announce_outputs(stage: str, paths: List[str]):
    targets = [p for p in paths if p]
    if targets:
        logger.info("ℹ️ Stage %s will modify/append: %s", stage, ", ".join(targets))

def main():
    parser = argparse.ArgumentParser(description="Memory build/trace/retrieve/generate pipeline")
    parser.add_argument(
        "--stage",
        choices=["build", "trace", "retrieve", "generate", "all"],
        default="all",
        help="Which stage to run; 'all' runs the full pipeline sequentially",
    )
    parser.add_argument("--build-prev-msgs", type=int, default=Config.BUILD_PREV_MSGS, help="How many previous messages to use when deciding splits")
    parser.add_argument("--answer-topn", type=str, default=str(Config.ANSWER_TOP_N), help="Number of boxes to use when answering (comma-separated)")
    parser.add_argument(
        "--text-modes",
        nargs="+",
        choices=["content", "content_trace_event", "trace_event"],
        help="Text modes for generation; default content only",
    )
    parser.add_argument("--run-id", type=str, help="Run identifier (defaults to model name)")
    args = parser.parse_args()

    Config.apply_run_id(args.run_id)
    Config.BUILD_PREV_MSGS = max(1, args.build_prev_msgs)
    Config.TOP_K_RETRIEVE = None  # Keep full ranking
    
    try:
        topn_list = [int(x) for x in args.answer_topn.split(",")]
    except ValueError:
        topn_list = [int(args.answer_topn)]
    Config.ANSWER_TOP_N = topn_list[0]

    worker = LLMWorker()

    text_modes = args.text_modes or Config.GEN_TEXT_MODES
    needs_trace = "content_trace_event" in text_modes or "trace_event" in text_modes

    logger.info("ℹ️ Using run_id=%s, output_dir=%s", Config.RUN_ID, Config.OUTPUT_DIR)

    if args.stage in ("build", "all"):
        _announce_outputs("build", [Config.FINAL_CONTENT_FILE, Config.BUILD_TRACE_FILE, Config.TOKEN_LOG_FILE, Config.BUILD_STATS_FILE, Config.VECTOR_DIR])
        TokenAnalyzer.stage_stats["build"] = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}
        builder = MemoryBuilder(worker)
        boxes = builder.build_all()
        if not Config.CHECKPOINT_EVERY_SAMPLE:
            builder.save(boxes)
        builder.summarize_and_log()
        logger.info("✅ Build done")

    if args.stage in ("trace", "all"):
        if needs_trace:
            _announce_outputs("trace", [Config.TIME_TRACE_FILE, Config.TRACE_PROMPT_LOG_FILE, Config.VECTOR_DIR])
            linker = TraceLinker(worker, trace_metrics=Config.TRACE_METRICS)
            linker.run()
        else:
            logger.info("ℹ️ Trace skipped because text_mode excludes trace events.")

    if args.stage in ("retrieve", "all"):
        _announce_outputs("retrieve", [Config.SIMPLE_RETRIEVAL_JSONL, Config.SIMPLE_RETRIEVAL_CSV, Config.VECTOR_DIR])
        retr = SimpleRetriever(worker, top_k=Config.TOP_K_RETRIEVE)
        retr.run(Config.SIMPLE_RETRIEVAL_JSONL, Config.SIMPLE_RETRIEVAL_CSV)

    if args.stage in ("generate", "all"):
        _announce_outputs("generate", [Config.GENERATION_RESULT_FILE, Config.GENERATION_REPORT_CSV, Config.GEN_SUMMARY_FILE, Config.TOKEN_LOG_FILE])

        if needs_trace and not os.path.exists(Config.TIME_TRACE_FILE):
            logger.warning("Trace file missing but trace text_mode requested; run trace stage first.")

        generator = AnswerGenerator(
            worker,
            answer_topn=topn_list,
            text_modes=text_modes,
            stage_label="gen",
        )
        generator.run(
            Config.SIMPLE_RETRIEVAL_JSONL,
            Config.GENERATION_RESULT_FILE,
            Config.GENERATION_REPORT_CSV,
        )


if __name__ == "__main__":
    main()
