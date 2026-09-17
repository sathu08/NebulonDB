"""
Client-side record schema for the bulk (MANY) ingestion pipelines.

Kept intentionally dependency-light so the client package can run against a
remote NebulonDB server without importing the server codebase.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParallelRecord(BaseModel):
    """One record flowing through an ingest/update pipeline.

    ``text`` is embedded server-side unless ``vector`` is precomputed.
    ``record_id`` is required for update/delete pipelines.
    Other metadata keys are stored verbatim under ``metadata``.
    """

    record_id: int | None = Field(default=None)
    text: str | None = Field(default=None)
    vector: list[float] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_precomputed(self) -> bool:
        return self.vector is not None

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"metadata": self.metadata}
        if self.record_id is not None:
            out["record_id"] = self.record_id
        if self.text is not None:
            out["text"] = self.text
        if self.vector is not None:
            out["vector"] = self.vector
        return out


def make_records(dataset: dict[str, list[Any]] | list[dict[str, Any]],
                 text_columns: list[str] | None = None) -> list[ParallelRecord]:
    """Convert a dict-of-lists (or list-of-dicts) dataset into records.

    A ``record_id`` key (present in either form) is lifted onto the model's
    ``record_id`` attribute so update/delete pipelines can address rows.
    """

    def _build(row: dict[str, Any], text_columns: list[str]) -> ParallelRecord:
        meta = dict(row)
        rid = meta.pop("record_id", None)
        text = ""
        for col in text_columns:
            t = meta.pop(col, None)
            if t:
                text = str(t)
                break
        return ParallelRecord(
            record_id=int(rid) if rid is not None else None,
            text=text or None,
            metadata=meta,
        )

    records: list[ParallelRecord] = []
    text_columns = text_columns or []

    if isinstance(dataset, list):
        for row in dataset:
            records.append(_build(row, text_columns))
        return records

    if isinstance(dataset, dict):
        if not dataset:
            return records
        sizes = {len(v) for v in dataset.values() if isinstance(v, (list, tuple))}
        n = max(sizes) if sizes else 0
        for i in range(n):
            row = {k: (v[i] if i < len(v) else None) for k, v in dataset.items()}
            records.append(_build(row, text_columns))
        return records

    raise TypeError(f"Unsupported dataset type: {type(dataset)}")