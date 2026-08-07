"""
Simple data containers used internally for index entries and compaction.
"""

from dataclasses import dataclass

@dataclass(slots=True)
class IndexEntry:
    segment_id: int
    offset: int
    version: int

@dataclass(slots=True)
class CompactionEntry:
    version: int
    payload: bytes