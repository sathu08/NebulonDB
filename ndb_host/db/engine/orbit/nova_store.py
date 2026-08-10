"""
NebulonDB Nova Persistent Store
================================

Storage layer bridging NovaEngine vectors to NebulonCosmos, containing:
    NovaStore             – CRUD interface for vector records with upsert semantics
    Record Retrieval      – single-doc and full-segment read operations
    Metadata Handling     – optional metadata attachment per vector record
"""


from typing import Optional, List, Dict, Any, Sequence

from db.engine import NebulonCosmos
from utils.logger import NebulonDBLogger

from db.engine.utils import (
    FIELD_ID,
    FIELD_VECTOR,
    FIELD_METADATA,
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
        metadata: Optional[Dict] = None,
        created_at: Optional[str] = None,
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

    def get(self, record_id: int) -> Optional[Dict[str, Any]]:
        return _rename_id(self._store.get_by_id(self.segment_name, record_id))

    def delete(self, record_id: int) -> int:
        return self._store.delete(self.segment_name, record_id)

    def read_all(self) -> List[Dict[str, Any]]:
        for rec in self._store.read_all(segment=self.segment_name, include_internal=True):
            yield _rename_id(dict(rec))

    def count(self) -> int:
        return len(list(self.read_all()))
