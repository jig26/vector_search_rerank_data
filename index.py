import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector-search")

app = FastAPI(title="Two-Stage Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent

# ---------------- Data loading (once, at import time) ---------------- #


def _load_documents() -> Dict[str, Dict[str, Any]]:
    """Load documents.csv into a doc_id -> row dict, with year cast to int."""
    docs = {}
    with open(DATA_DIR / "documents.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("doc_id")
            if not doc_id:
                continue
            row = dict(row)
            try:
                row["year"] = int(row["year"])
            except (TypeError, ValueError):
                pass  # leave as-is if not a clean int
            docs[doc_id] = row
    return docs


def _load_embeddings() -> Dict[str, List[float]]:
    with open(DATA_DIR / "embeddings.json", encoding="utf-8") as f:
        return json.load(f)


def _load_reranker_scores() -> Dict[str, Dict[str, float]]:
    with open(DATA_DIR / "reranker_scores.json", encoding="utf-8") as f:
        return json.load(f)


DOCUMENTS: Dict[str, Dict[str, Any]] = _load_documents()
EMBEDDINGS: Dict[str, List[float]] = _load_embeddings()
RERANKER_SCORES: Dict[str, Dict[str, float]] = _load_reranker_scores()

logger.info(
    "Loaded %d documents, %d embeddings, %d reranker score tables",
    len(DOCUMENTS), len(EMBEDDINGS), len(RERANKER_SCORES),
)

# ---------------- Models ---------------- #


class SearchRequest(BaseModel):
    query_id: Optional[str] = None
    query_vector: Optional[List[float]] = None
    top_k: Optional[int] = 10
    rerank_top_n: Optional[int] = 3
    filter: Optional[Dict[str, Any]] = None


# ---------------- Filtering ---------------- #


def _normalize(value: Any) -> Any:
    """Normalize a value for comparison: numeric-looking values become
    numbers (int if exact, else float) so that '2024' (csv string) and
    2024 (JSON int) compare equal; everything else is compared as a
    lowercased string so casing differences in categorical fields like
    department/region don't cause spurious filter misses."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            f = float(value)
            return int(f) if f.is_integer() else f
        except ValueError:
            return value.strip().lower()
    return value


def _matches_condition(doc_value: Any, condition: Any) -> bool:
    if isinstance(condition, dict):
        norm_doc = _normalize(doc_value)
        for op, target in condition.items():
            if op == "gte":
                if norm_doc is None or norm_doc < _normalize(target):
                    return False
            elif op == "lte":
                if norm_doc is None or norm_doc > _normalize(target):
                    return False
            elif op == "gt":
                if norm_doc is None or norm_doc <= _normalize(target):
                    return False
            elif op == "lt":
                if norm_doc is None or norm_doc >= _normalize(target):
                    return False
            elif op == "in":
                targets = {_normalize(t) for t in (target or [])}
                if norm_doc not in targets:
                    return False
            elif op == "eq":
                if norm_doc != _normalize(target):
                    return False
            else:
                # Unknown operator: ignore rather than fail the whole request
                continue
        return True
    else:
        return _normalize(doc_value) == _normalize(condition)


def apply_filter(documents: Dict[str, Dict[str, Any]], filters: Optional[Dict[str, Any]]):
    if not filters:
        return list(documents.values())
    matched = []
    for doc in documents.values():
        ok = True
        for field, condition in filters.items():
            if field not in doc:
                ok = False
                break
            if not _matches_condition(doc[field], condition):
                ok = False
                break
        if ok:
            matched.append(doc)
    return matched


# ---------------- Similarity ---------------- #


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------- Routes ---------------- #


@app.get("/")
def home():
    return {"status": "running", "documents_loaded": len(DOCUMENTS)}


@app.post("/vector-search")
async def vector_search(request: Request):
    try:
        raw = await request.json()
    except Exception:
        logger.info("REQUEST_BODY_UNPARSEABLE")
        return JSONResponse(content={"matches": []})

    logger.info("INCOMING_REQUEST: %s", json.dumps(raw)[:2000])

    try:
        req = SearchRequest(**raw) if isinstance(raw, dict) else SearchRequest()
    except Exception:
        logger.info("REQUEST_VALIDATION_FAILED")
        return JSONResponse(content={"matches": []})

    query_vector = req.query_vector or []
    top_k = req.top_k if isinstance(req.top_k, int) and req.top_k > 0 else 10
    rerank_top_n = (
        req.rerank_top_n if isinstance(req.rerank_top_n, int) and req.rerank_top_n > 0 else 3
    )

    try:
        # ---- Stage 0: metadata filter ----
        candidates = apply_filter(DOCUMENTS, req.filter)

        # ---- Stage 1: vector similarity, top_k ----
        scored = []
        for doc in candidates:
            doc_id = doc["doc_id"]
            emb = EMBEDDINGS.get(doc_id)
            if emb is None:
                continue
            sim = cosine_similarity(query_vector, emb)
            scored.append((doc_id, sim))

        scored.sort(key=lambda x: (-x[1], x[0]))
        stage1_ids = [doc_id for doc_id, _ in scored[:top_k]]

        # ---- Stage 2: re-rank via lookup table, rerank_top_n ----
        rerank_table = RERANKER_SCORES.get(req.query_id, {}) if req.query_id else {}

        reranked = []
        for doc_id in stage1_ids:
            score = rerank_table.get(doc_id)
            if score is None:
                # No rerank score available for this query/doc pair -- fall
                # back to stage-1 similarity so a missing lookup entry
                # doesn't crash the request or silently drop the doc.
                score = dict(scored).get(doc_id, -1.0)
            reranked.append((doc_id, score))

        reranked.sort(key=lambda x: (-x[1], x[0]))
        matches = [doc_id for doc_id, _ in reranked[:rerank_top_n]]

        logger.info(
            "OUTGOING_RESPONSE for query_id=%r: matches=%s",
            req.query_id, matches,
        )
        return JSONResponse(content={"matches": matches})

    except Exception:
        logger.exception("VECTOR_SEARCH_FAILED")
        return JSONResponse(content={"matches": []})