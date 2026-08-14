"""
NebulonDB Nova Persistent Store
================================

Storage layer bridging NovaEngine vectors to NebulonCosmos, containing:
    NovaStore             – CRUD interface for vector records with upsert semantics
    Record Retrieval      – single-doc and full-segment read operations
    Metadata Handling     – optional metadata attachment per vector record
"""


from typing import Any
from collections.abc import Sequence

from db.engine import NebulonCosmos

from db.engine.utils import (
    FIELD_ID,
    FIELD_VECTOR,
    FIELD_CREATED_AT,
)


def _rename_id(doc):
    """Surface the internal cosmos key as public 'id'."""
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = doc.pop("_id")
    return doc


class NovaStore:
    """Persistent store for vector records.

    Each row on disk carries only the vector fingerprint for the entity:
        {id, vector: float[], created_at}
    Documents and metadata are kept separately in nebulo_documents.
    """

    def __init__(self, store: NebulonCosmos, segment_name: str):
        self._store = store
        self.segment_name = segment_name

    def insert(
        self,
        record_id: int,
        vector: Sequence[float],
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> int:
        doc = {
            FIELD_ID: record_id,
            FIELD_VECTOR: list(vector),
            FIELD_CREATED_AT: created_at,
        }
        existing = self._store.get_by_id(self.segment_name, record_id)
        if existing is None:
            return self._store.insert(self.segment_name, doc)
        self._store.update(self.segment_name, doc)
        return record_id

    def get(self, record_id: int) -> dict[str, Any] | None:
        return _rename_id(self._store.get_by_id(self.segment_name, record_id))

    def delete(self, record_id: int) -> int:
        return self._store.delete(self.segment_name, record_id)

    def read_all(self) -> list[dict[str, Any]]:
        for rec in self._store.read_all(segment=self.segment_name, include_internal=True):
            yield _rename_id(dict(rec))

    def count(self) -> int:
        return len(list(self.read_all()))
