"""
NebulonDB Document Store
========================

Session store for the `nebulon_documents` segment. Keeps the human-facing
document material (text + metadata) separate from the Nova vector rows and
the Mesh node/edge rows.

Each row on disk:
    {id, text, metadata{label, lang, type, retention, expires_at}, created_at}
"""


from typing import Any

from db.engine import NebulonCosmos
from utils.logger import NebulonDBLogger

from db.engine.utils import (
    FIELD_ID,
    FIELD_TEXT,
    FIELD_LABEL,
    FIELD_METADATA,
    FIELD_CREATED_AT,
)

logger = NebulonDBLogger().get_logger()

# The persisted Cosmos index is keyed by bare record_id while the id space is
# global across all tables. Document rows must live in a disjoint id range so
# they never collide with Nova (identity) or Mesh rows in the index.
ID_OFFSET = 1 << 40


def _rename_id(doc):
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = doc.pop("_id") - ID_OFFSET
    return doc


class DocumentStore:
    def __init__(self, store: NebulonCosmos, segment_name: str):
        self._store = store
        self.segment_name = segment_name

    def insert(
        self,
        record_id: int,
        text: str,
        metadata: dict | None = None,
        label: str | None = None,
        created_at: str | None = None,
    ) -> int:
        doc = {
            FIELD_ID: record_id + ID_OFFSET,
            FIELD_TEXT: text,
            FIELD_METADATA: dict(metadata or {}),
            FIELD_CREATED_AT: created_at,
        }
        if label is not None:
            doc.setdefault(FIELD_METADATA, {}).setdefault(FIELD_LABEL, label)
        existing = self._store.get_by_id(self.segment_name, record_id + ID_OFFSET)
        if existing is None:
            return self._store.insert(self.segment_name, doc)
        self._store.update(self.segment_name, doc)
        return record_id

    def update_metadata(self, record_id: int, metadata: dict[str, Any]) -> int:
        existing = self._store.get_by_id(self.segment_name, record_id + ID_OFFSET)
        if existing is None:
            return 0
        merged = dict(existing.get(FIELD_METADATA) or {})
        merged.update(metadata)
        existing[FIELD_METADATA] = merged
        return self._store.update(self.segment_name, existing)

    def get(self, record_id: int) -> dict[str, Any] | None:
        return _rename_id(self._store.get_by_id(self.segment_name, record_id + ID_OFFSET))

    def delete(self, record_id: int) -> int:
        return self._store.delete(self.segment_name, record_id + ID_OFFSET)

    def delete_many(self, record_ids: list[int]) -> list[int]:
        """Bulk-delete several documents in a single WAL + memtable pass."""
        if not record_ids:
            return []
        ids = [rid + ID_OFFSET for rid in record_ids]
        return self._store.delete_many(self.segment_name, ids)

    def read_all(self) -> list[dict[str, Any]]:
        return [
            _rename_id(dict(rec))
            for rec in self._store.read_all(segment=self.segment_name, include_internal=True)
        ]
