"""Upstash Vector instruction embedding index for NavBuddy.

Builds and queries a cloud-hosted vector index of ``enhanced_instruction``
strings from inference results.  Upstash handles embedding internally using
the model configured on the index (bge-small-en-v1.5, 384-dim, cosine).

Use cases:
- Semantic search across all generated instructions
- DPO hard negative mining (find responses similar to rejected but not chosen)
- Detecting near-duplicate outputs across models

Usage::

    from navbuddy.eval.embeddings import InstructionIndex

    idx = InstructionIndex.from_env()
    idx.index_jsonl("results/results_gemini.jsonl")

    # Semantic search
    hits = idx.find_similar("Turn left at the traffic lights", n=5)
    for h in hits:
        print(h.score, h.metadata["sample_id"], h.metadata.get("text", "")[:80])

    # DPO hard negatives
    neg_ids = idx.find_hard_negatives(chosen_text, rejected_text, n=5)

Environment variables::

    UPSTASH_VECTOR_REST_URL    – Upstash Vector REST endpoint
    UPSTASH_VECTOR_REST_TOKEN  – Upstash Vector REST token
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InstructionIndex:
    """Upstash Vector index of enhanced_instruction embeddings.

    The index must be pre-configured in the Upstash console with:
        Model  : bge-base-en-v1.5
        Dims   : 768
        Metric : cosine

    Upstash embeds text server-side — no local model required.

    Args:
        url: Upstash Vector REST URL.
        token: Upstash Vector REST token.
        namespace: Optional namespace within the index (default: "instructions").
    """

    def __init__(self, url: str, token: str, namespace: str = "instructions"):
        try:
            from upstash_vector import Index
        except ImportError as exc:
            raise RuntimeError(
                "InstructionIndex requires upstash-vector. "
                "Install: pip install upstash-vector"
            ) from exc

        self._index = Index(url=url, token=token)
        self._namespace = namespace or ""
        logger.info("InstructionIndex connected to Upstash Vector (%s)", url)

    @classmethod
    def from_env(cls, namespace: str = "instructions") -> "InstructionIndex":
        """Create from UPSTASH_VECTOR_REST_URL / UPSTASH_VECTOR_REST_TOKEN env vars."""
        url = os.environ.get("UPSTASH_VECTOR_REST_URL", "")
        token = os.environ.get("UPSTASH_VECTOR_REST_TOKEN", "")
        if not url or not token:
            raise RuntimeError(
                "Set UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN "
                "environment variables to use InstructionIndex."
            )
        return cls(url=url, token=token, namespace=namespace)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_results(self, results: Dict[str, Any]) -> int:
        """Index enhanced_instructions from a DataManager.results dict.

        Args:
            results: Mapping of ``sample_id → {result_key → result_dict}``.

        Returns:
            Number of vectors upserted.
        """
        vectors = []
        for sample_id, models in results.items():
            if not isinstance(models, dict):
                continue
            for result_key, result in models.items():
                if not isinstance(result, dict):
                    continue
                text = result.get("enhanced_instruction", "")
                if not text:
                    continue
                vectors.append({
                    "id": f"{sample_id}::{result_key}",
                    "data": text,
                    "metadata": {
                        "sample_id": str(sample_id),
                        "result_key": str(result_key),
                        "text": text[:500],
                    },
                })

        return self._upsert_batch(vectors)

    def index_jsonl(self, results_jsonl: str) -> int:
        """Index from a results JSONL file (one InferenceResult JSON per line).

        Args:
            results_jsonl: Path to results JSONL file.

        Returns:
            Number of vectors upserted.
        """
        vectors = []
        with open(results_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                text = r.get("enhanced_instruction", "")
                if not text:
                    continue
                uid = "{}::{}::{}".format(
                    r.get("sample_id", ""),
                    r.get("model_id", "").replace("/", "_"),
                    r.get("modality", "").replace(" ", "_"),
                )
                vectors.append({
                    "id": uid,
                    "data": text,
                    "metadata": {
                        "sample_id": str(r.get("sample_id", "")),
                        "model_id": str(r.get("model_id", "")),
                        "modality": str(r.get("modality", "")),
                        "augment": str(r.get("augment") or ""),
                        "text": text[:500],
                    },
                })

        return self._upsert_batch(vectors)

    def _upsert_batch(self, vectors: list, batch_size: int = 100, namespace: Optional[str] = None) -> int:
        """Upsert vectors in batches (Upstash max per request is 1000)."""
        if not vectors:
            return 0
        ns = namespace if namespace is not None else self._namespace
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=ns)
            total += len(batch)
        logger.info("InstructionIndex: upserted %d vectors (namespace=%s)", total, ns)
        return total

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def find_similar(self, text: str, n: int = 10) -> list:
        """Return the n most similar instructions to text.

        Returns:
            List of QueryResult objects with ``.id``, ``.score``, ``.metadata``.
        """
        return self._index.query(data=text, top_k=n, include_metadata=True, namespace=self._namespace)

    def find_hard_negatives(
        self,
        chosen_text: str,
        rejected_text: str,
        n: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find instructions similar to rejected but not to chosen (hard negatives for DPO).

        Args:
            chosen_text: The preferred instruction text.
            rejected_text: The rejected instruction text.
            n: Number of hard negatives to return.

        Returns:
            List of (vector_id, similarity_score) tuples. Score is cosine similarity
            to the rejected text — higher means more confusingly similar to the rejected.
        """
        pool = max(n * 4, 20)
        similar_rejected = self._index.query(data=rejected_text, top_k=pool, include_metadata=True, namespace=self._namespace)
        similar_chosen = self._index.query(data=chosen_text, top_k=n, include_metadata=True, namespace=self._namespace)

        chosen_ids = {r.id for r in similar_chosen}

        hard_negatives: List[Tuple[str, float]] = []
        for r in similar_rejected:
            if r.id not in chosen_ids:
                hard_negatives.append((r.id, r.score))
            if len(hard_negatives) >= n:
                break

        return hard_negatives

    # ------------------------------------------------------------------
    # GT landmark indexing
    # ------------------------------------------------------------------

    LANDMARK_NAMESPACE = "gt_landmarks"

    def index_gt_landmarks(
        self,
        gt_jsonl_paths: List[str],
        namespace: str = LANDMARK_NAMESPACE,
    ) -> int:
        """Index GT landmark annotations (positive/negative) into Upstash Vector.

        Reads gt_fields.jsonl and/or custom_labels.jsonl, deduplicates by sample_id
        keeping the most recent entry, and indexes the enhanced_instruction text so
        that find_nearest_annotated() can retrieve landmark context for unannotated
        samples via semantic similarity.

        Metadata stored per vector (landmark lists as JSON strings — Upstash metadata
        must be flat scalars):
            sample_id, positive_landmarks (JSON), negative_landmarks (JSON),
            annotator, source (filename), text (first 500 chars)

        Args:
            gt_jsonl_paths: Paths to annotation JSONL files (gt_fields, custom_labels).
            namespace: Upstash namespace (default: "gt_landmarks").

        Returns:
            Number of vectors upserted.
        """
        # Collect entries keyed by sample_id, keeping latest updated_at per sample.
        entries: Dict[str, dict] = {}
        for path in gt_jsonl_paths:
            if not os.path.exists(path):
                logger.warning("GT JSONL not found: %s", path)
                continue
            source = os.path.basename(path)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    sid = r.get("sample_id", "")
                    if not sid:
                        continue
                    pos = r.get("positive_landmarks") or []
                    neg = r.get("negative_landmarks") or []
                    if not pos and not neg:
                        continue  # skip entries with no landmark annotations
                    text = r.get("enhanced_instruction", "")
                    if not text:
                        continue
                    ts = r.get("updated_at", r.get("timestamp", ""))
                    existing = entries.get(sid)
                    if existing is None or ts > existing.get("updated_at", ""):
                        entries[sid] = {
                            "sample_id": sid,
                            "enhanced_instruction": text,
                            "positive_landmarks": pos,
                            "negative_landmarks": neg,
                            "annotator": r.get("annotator", ""),
                            "updated_at": ts,
                            "source": source,
                        }

        vectors = []
        for sid, e in entries.items():
            text = e["enhanced_instruction"]
            vectors.append({
                "id": f"gt_landmark::{sid}",
                "data": text,
                "metadata": {
                    "sample_id": sid,
                    "positive_landmarks": json.dumps(e["positive_landmarks"]),
                    "negative_landmarks": json.dumps(e["negative_landmarks"]),
                    "annotator": str(e["annotator"]),
                    "source": e["source"],
                    "text": text[:500],
                },
            })

        if not vectors:
            logger.warning("index_gt_landmarks: no annotated samples found in %s", gt_jsonl_paths)
            return 0

        return self._upsert_batch(vectors, namespace=namespace)

    def find_nearest_annotated(
        self,
        instruction_text: str,
        n: int = 3,
        namespace: str = LANDMARK_NAMESPACE,
    ) -> list:
        """Find the nearest GT-annotated samples to a given instruction.

        Args:
            instruction_text: Instruction text to query with.
            n: Number of results to return.
            namespace: Landmark namespace (default: "gt_landmarks").

        Returns:
            List of QueryResult objects with .id, .score, .metadata
            (metadata includes positive_landmarks and negative_landmarks as JSON strings).
        """
        return self._index.query(
            data=instruction_text,
            top_k=n,
            include_metadata=True,
            namespace=namespace,
        )

    def count(self) -> int:
        """Return approximate number of indexed vectors."""
        try:
            info = self._index.info()
            return info.vector_count
        except Exception:
            return -1

    def delete(self, *ids: str) -> None:
        """Delete vectors by ID."""
        self._index.delete(ids=list(ids), namespace=self._namespace)
