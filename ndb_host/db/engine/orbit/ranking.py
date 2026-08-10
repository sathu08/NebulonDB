"""
NebulonDB Complete Ranking & Re‑Ranking Engine
===============================================

Orbit package module containing:
    BM25Scorer            – scalable BM25 index (keyed by doc ID)
    RRFMerger             – Reciprocal Rank Fusion for HNSW + BM25 lists
    QueryIntent           – dynamic weight selector
    RankEngine            – multi‑signal rank fusion (normalised + optional RRF)
    CrossEncoderReranker  – lazy‑loaded cross‑encoder
"""

from __future__ import annotations

import math

from dataclasses import dataclass

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Callable

from ndb_host.utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()


# ----------------------------------------------------------------------
# 0. Utility helpers
# ----------------------------------------------------------------------
def sigmoid(x: float, k: float = 1.0) -> float:
    """Squash any real value to (0,1). k controls steepness."""
    return 1.0 / (1.0 + math.exp(-k * x))


# ----------------------------------------------------------------------
# 0.5 Ranking configuration
# ----------------------------------------------------------------------
@dataclass
class RankConfig:
    """Holds ranking / re-ranking settings.

    Configured once (at NebulonOrbit construction time) and consumed by
    ``search`` / ``ranked_search`` / ``rerank`` instead of being passed as
    per-call parameters.
    """

    use_rrf: bool = True
    rerank: bool = False
    weights: Optional[Dict[str, float]] = None
    half_life: float = 30.0
    metadata_rules: Optional[Callable[[Dict[str, Any]], float]] = None


# ----------------------------------------------------------------------
# 1. Scalable BM25 Index
# ----------------------------------------------------------------------
class BM25Scorer:
    """
    Pre‑built BM25 index over a document corpus.

    Args:
        documents: list of dicts with keys 'id' and 'text'.
        k1, b: BM25 hyperparameters.
    """

    def __init__(self, documents: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = [doc["id"] for doc in documents]
        self.tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_corpus]
        self.n_docs = len(self.doc_ids)
        self.avg_doc_len = sum(self.doc_lengths) / self.n_docs if self.n_docs else 0
        self.idf_cache: Dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return str(text or "").lower().split()

    def _idf(self, term: str) -> float:
        if term not in self.idf_cache:
            df = sum(1 for tokens in self.tokenized_corpus if term in tokens)
            self.idf_cache[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return self.idf_cache[term]

    def score(self, query: str, doc_id: Union[str, int]) -> float:
        """BM25 score for a single document (raw value)."""
        try:
            idx = self.doc_ids.index(doc_id)
        except ValueError:
            return 0.0
        doc_tokens = self.tokenized_corpus[idx]
        doc_len = self.doc_lengths[idx]
        query_tokens = self._tokenize(query)
        term_freqs = {t: doc_tokens.count(t) for t in query_tokens}
        score = 0.0
        for term in set(query_tokens):
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Full BM25 search returning top_k documents with their raw BM25 scores.
        Format: [{"id": doc_id, "score": bm25_score, "text": ...}, ...]
        """
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in self.doc_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]
        # Build enriched results (the caller can add metadata later)
        id_to_text = {self.doc_ids[i]: self._untokenize(self.tokenized_corpus[i]) for i in range(self.n_docs)}
        results = []
        for doc_id, sc in top:
            results.append({
                "id": doc_id,
                "score": sc,
                "text": id_to_text.get(doc_id, ""),
            })
        return results

    @staticmethod
    def _untokenize(tokens: List[str]) -> str:
        return " ".join(tokens)


# ----------------------------------------------------------------------
# 2. Reciprocal Rank Fusion (RRF) Merger
# ----------------------------------------------------------------------
class RRFMerger:
    """
    Fuses two result lists (e.g., HNSW and BM25) using Reciprocal Rank Fusion.

    Args:
        k: constant (default 60) to smooth rankings.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def merge(
        self,
        list_a: List[Dict[str, Any]],
        list_b: List[Dict[str, Any]],
        id_key: str = "id",
        max_unique: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Combine two lists, each containing dicts with at least an 'id' field.
        Returns a new list of unique documents with an 'rrf_score' field,
        sorted descending.
        """
        rrf = {}
        doc_cache = {}  # keep first seen full dict

        for rank, item in enumerate(list_a):
            rid = item[id_key]
            rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (self.k + rank + 1)
            if rid not in doc_cache:
                doc_cache[rid] = dict(item)

        for rank, item in enumerate(list_b):
            rid = item[id_key]
            rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (self.k + rank + 1)
            if rid not in doc_cache:
                doc_cache[rid] = dict(item)

        merged = []
        for rid, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
            if len(merged) >= max_unique:
                break
            entry = doc_cache[rid]
            entry["rrf_score"] = score
            merged.append(entry)

        return merged


# ----------------------------------------------------------------------
# 3. Query Intent Classifier
# ----------------------------------------------------------------------
class QueryIntent:
    """
    Simple keyword‑based intent detector that returns a weight dict.
    Override with your own NLP classifier for production.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights

    def get_weights(self, query: str) -> Dict[str, float]:
        q = query.lower()
        if any(w in q for w in ["latest", "recent", "new", "today", "update"]):
            return {"vector": 0.45, "bm25": 0.20, "metadata": 0.10, "importance": 0.05, "freshness": 0.20}
        if any(w in q for w in ["definition", "explain", "what is", "architecture"]):
            return {"vector": 0.60, "bm25": 0.25, "metadata": 0.05, "importance": 0.05, "freshness": 0.05}
        if self.weights is not None:
            return self.weights
        return {"vector": 0.55, "bm25": 0.20, "metadata": 0.10, "importance": 0.10, "freshness": 0.05}


# ----------------------------------------------------------------------
# 4. Rank Engine (multi‑signal fusion)
# ----------------------------------------------------------------------
class RankEngine:
    """
    Combines up to five normalised signals into a final rank score.

    Supports two fusion modes:
      - 'linear' (default): weighted sum of all signals (all in [0,1]).
      - 'rrf' : only used when candidates come pre‑merged via RRF (see SearchPipeline).
                In this mode, vector+bm25 are replaced by rrf_score;
                the other signals (metadata, importance, freshness) are added.

    All signals are normalised to [0,1] before fusion.
    """

    def __init__(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
        half_life: float = 30.0,
        mode: str = "linear",
    ):
        self.mode = mode
        self.half_life = half_life
        self.weights = weights
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            logger.warning("Weights sum to %s, not 1.0; normalizing", total)
            if total:
                self.weights = {k: v / total for k, v in self.weights.items()}
            else:
                logger.error("All weights are zero; falling back to defaults")
                self.weights = {
                    "vector": 0.55, "bm25": 0.20, "metadata": 0.10,
                    "importance": 0.10, "freshness": 0.05,
                }

        self.bm25 = BM25Scorer(documents) if documents and mode == "linear" else None
        self.metadata_rules = self._default_metadata_rules

    def _default_metadata_rules(self, meta: Dict[str, Any]) -> float:
        score = 0.0
        doc_type = str(meta.get("type") or "").lower()
        lang = str(meta.get("lang") or "").lower()
        if doc_type in ("pdf", "markdown", "notebook", "important_chat"):
            score += 0.3
        if lang == "en":
            score += 0.1
        if meta.get("retention") == "permanent":
            score += 0.2
        return min(score, 1.0)

    def _vector_score(self, candidate: dict) -> float:
        raw = candidate.get("score", 0.0)
        return sigmoid(raw, k=5.0)

    def _bm25_score(self, query: str, candidate: dict) -> float:
        if self.bm25 is None:
            return 0.0
        doc_id = candidate.get("id")
        if doc_id is None:
            return 0.0
        raw = self.bm25.score(query, doc_id)
        return sigmoid(raw, k=0.5)
    def _metadata_score(self, candidate: dict) -> float:
        meta = candidate.get("metadata", candidate)
        return self.metadata_rules(meta)

    def _importance_score(self, candidate: dict) -> float:
        meta = candidate.get("metadata", {}) or {}
        val = candidate.get("importance", meta.get("importance", 0.0))
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _freshness_score(self, candidate: dict) -> float:
        meta = candidate.get("metadata", {}) or {}
        ts = (candidate.get("created_at") or candidate.get("timestamp")
              or meta.get("created_at") or meta.get("timestamp"))
        if ts is None:
            return 0.0
        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return 0.0
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return 2 ** (-age_days / self.half_life)

    def rank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        return_top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates. In 'linear' mode, all signals are combined.
        In 'rrf' mode, vector+bm25 are replaced by the pre‑computed 'rrf_score'
        (assumed already normalised, typically from RRFMerger).
        """
        scored = []
        for cand in candidates:
            cand = dict(cand)

            if self.mode == "rrf":
                rrf = cand.get("rrf_score", 0.0)
                vs = 0.0
                bs = 0.0
            else:
                vs = self._vector_score(cand)
                bs = self._bm25_score(query, cand)
                rrf = 0.0

            ms = self._metadata_score(cand)
            im = self._importance_score(cand)
            fs = self._freshness_score(cand)

            if self.mode == "rrf":
                rrf_weight = self.weights.get("vector", 0.55) + self.weights.get("bm25", 0.20)
                final = (rrf_weight * rrf
                         + self.weights.get("metadata", 0.10) * ms
                         + self.weights.get("importance", 0.10) * im
                         + self.weights.get("freshness", 0.05) * fs)
            else:
                final = (self.weights.get("vector", 0.55) * vs
                         + self.weights.get("bm25", 0.20) * bs
                         + self.weights.get("metadata", 0.10) * ms
                         + self.weights.get("importance", 0.10) * im
                         + self.weights.get("freshness", 0.05) * fs)

            cand["rank_score"] = final
            cand["_rank_debug"] = {
                "vector": vs,
                "bm25": bs,
                "rrf": rrf,
                "metadata": ms,
                "importance": im,
                "freshness": fs,
            }
            scored.append(cand)

        scored.sort(key=lambda x: x["rank_score"], reverse=True)
        if return_top_n is not None:
            scored = scored[:return_top_n]
        return scored


# ----------------------------------------------------------------------
# 5. Cross‑Encoder Re‑Ranker (lazy loaded)
# ----------------------------------------------------------------------
class CrossEncoderReranker:
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self._sem_model = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._sem_model is None:
            try:
                from core.model_hub import SemanticEmbeddingModel
            except ImportError as e:
                logger.warning(
                    "CrossEncoderReranker unavailable (model_hub not importable): %s. "
                    "Skipping cross-encoder re-rank.", e
                )
                self._available = False
                return None
            self._sem_model = SemanticEmbeddingModel()
            self._available = True
        return self._sem_model

    def available(self) -> bool:
        """Return True if the cross-encoder dependency can be loaded."""
        if self._available is None:
            self._load()
        return bool(self._available)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        sem_model = self._load()
        if sem_model is None:
            return documents
        pairs = [(query, doc.get(text_key, "")) for doc in documents]
        scores = sem_model.cross_encode(pairs, show_progress_bar=False)
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            documents = documents[:top_k]
        return documents
