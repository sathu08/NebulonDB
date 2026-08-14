"""
cosmos/compactor.py
===================
Compaction logic for NebulonCosmos.

Responsibilities
----------------
- `compact`                   – merge N segments into one, removing deleted records.
- `cleanup_segments`          – evict segments from caches (and optionally delete files).
- `remove_segments_and_rebuild` – delete segments and rebuild the index.
- `background_compaction_loop` – daemon thread body: triggers compact when needed.
"""

import threading
import contextlib
import zlib
import struct

from pathlib import Path

from db.engine.utils import BloomFilter
from utils.logger import NebulonDBLogger

from db.engine.utils import decode_object


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#  Segment cleanup helper
# ==========================================================


def cleanup_segments(
    fnames: list[str],
    seg_dir: Path,
    segment_cache: dict,
    segment_size_cache: dict[int, int],
    bloom_filter_cache: dict[int, BloomFilter],
    segment_info: dict[int, dict],
    delete_files: bool = True,
) -> None:
    """Remove segments from all caches, manifest, and optionally delete files."""
    for fname in fnames:
        seg_id = int(fname.split('_')[1].split('.')[0])
        if delete_files:
            with contextlib.suppress(FileNotFoundError):
                (seg_dir / fname).unlink()
        segment_size_cache.pop(seg_id, None)
        bloom_filter_cache.pop(seg_id, None)
        segment_info.pop(seg_id, None)
        if fname in segment_cache:
            f, mm = segment_cache.pop(fname)
            mm.close()
            f.close()


# ==========================================================
#  Remove segments and rebuild index
# ==========================================================

def remove_segments_and_rebuild(
    fnames,
    seg_dir, manifest, segment_cache,
    segment_size_cache, bloom_filter_cache, segment_info,
    save_manifest_and_meta_fn, rebuild_index_fn,
) -> None:
    for fname in fnames:
        seg_id = int(fname.split('_')[1].split('.')[0])
        with contextlib.suppress(FileNotFoundError):
            (seg_dir / fname).unlink()
        if fname in manifest:
            manifest.remove(fname)
        segment_size_cache.pop(seg_id, None)
        bloom_filter_cache.pop(seg_id, None)
        segment_info.pop(seg_id, None)
        if fname in segment_cache:
            f, mm = segment_cache.pop(fname)
            mm.close()
            f.close()

    manifest.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    save_manifest_and_meta_fn()
    rebuild_index_fn()


# ==========================================================
#  Main compaction routine
# ==========================================================

def compact(
    segments_to_merge: list[str] | None,
    manifest: list[str],
    seg_dir: Path,
    max_segments_before_compact: int,
    # readers
    validate_segment_fn,
    get_segment_data_offset_fn,
    get_next_segment_id_fn,
    write_segment_streaming_fn,
    rewrite_index_fn,
    segment_cache: dict,
    segment_size_cache: dict[int, int],
    bloom_filter_cache: dict[int, BloomFilter],
    segment_info: dict[int, dict],
    record_header_size: int,
    record_header_format: str,
    save_manifest_and_meta_fn,
) -> None:
    """
    Merge *segments_to_merge* into a single new segment, dropping deleted records.

    If *segments_to_merge* is None, the function decides automatically which
    segments to merge based on *max_segments_before_compact*.
    """
    if segments_to_merge is None:
        if len(manifest) <= max_segments_before_compact:
            return
        segments_to_merge = manifest[:len(manifest) - 3]
        if len(segments_to_merge) < 2:
            return

    if len(segments_to_merge) < 2:
        logger.info("Insufficient segments for compaction merge")
        return

    # ── Pass 1: find latest non-deleted record locations ─────
    # Key by (segment, rec_id) so records in different segments that happen to
    # share a record id are never silently dropped during the merge.
    latest_source: dict[tuple[str, int], tuple[str, int, int]] = {}
    for fname in sorted(
        segments_to_merge,
        key=lambda x: int(x.split('_')[1].split('.')[0]),
        reverse=True,
    ):
        seg_path = seg_dir / fname
        valid, count, bf_data, compressed = validate_segment_fn(seg_path)
        if not valid:
            continue
        data_offset = get_segment_data_offset_fn(
            seg_path, len(bf_data) if bf_data else 0
        )
        with seg_path.open("rb") as f:
            f.seek(data_offset)
            for _ in range(count):
                offset = f.tell()
                header_data = f.read(record_header_size)
                if len(header_data) != record_header_size:
                    break
                stored_crc, comp_len, _ = struct.unpack(record_header_format, header_data)
                comp_data = f.read(comp_len)
                if len(comp_data) != comp_len:
                    break
                if compressed:
                    try:
                        payload = zlib.decompress(comp_data)
                    except zlib.error:
                        continue
                else:
                    payload = comp_data
                if (zlib.crc32(payload) & 0xFFFFFFFF) != stored_crc:
                    continue
                try:
                    rec_dict = decode_object(payload)
                except Exception:
                    continue
                rec_id = rec_dict.get("_id")
                if rec_id is None:
                    continue
                segment = rec_dict.get("_segment") or "_main"
                skey = (segment, rec_id)
                if skey not in latest_source and not rec_dict.get("_deleted", False):
                    version = rec_dict.get("_version", 0)
                    latest_source[skey] = (fname, offset, version)

    if not latest_source:
        logger.info("No live records found, cleaning up merged segments")
        cleanup_segments(
            segments_to_merge, seg_dir,
            segment_cache, segment_size_cache, bloom_filter_cache, segment_info,
            delete_files=False,
        )
        rewrite_index_fn()
        return

    # ── Pass 2: write new segment ─────────────────────────────
    new_seg_id = get_next_segment_id_fn()
    new_seg_name = f"seg_{new_seg_id}.ndb"
    new_seg_path = seg_dir / new_seg_name

    write_segment_streaming_fn(new_seg_path, latest_source, new_seg_id, append_index=False)

    # ── Atomic manifest update ────────────────────────────────
    untouched = [f for f in manifest if f not in segments_to_merge]
    new_manifest_list = untouched + [new_seg_name]
    new_manifest_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    manifest.clear()
    manifest.extend(new_manifest_list)
    save_manifest_and_meta_fn()

    # ── Delete old segment files ──────────────────────────────
    cleanup_segments(
        segments_to_merge, seg_dir,
        segment_cache, segment_size_cache, bloom_filter_cache, segment_info,
        delete_files=True,
    )

    # ── Rewrite index ─────────────────────────────────────────
    rewrite_index_fn()

    logger.info(
        f"Compaction merged {len(segments_to_merge)} segments, "
        f"retained {len(latest_source)} active records"
    )


# ==========================================================
#  Background compaction daemon
# ==========================================================

def background_compaction_loop(
    stop_event: threading.Event,
    compaction_interval: float,
    manifest: list[str],
    max_segments_before_compact: int,
    compact_fn,     # () -> None (calls compact with no segments_to_merge arg)
    db_dir: Path | None = None,
) -> None:
    """
    Daemon thread body.

    Sleeps for *compaction_interval* seconds between checks.
    Calls *compact_fn* when the manifest is too long.

    If *db_dir* is provided and the directory is deleted (e.g. corpus
    teardown), the loop stops itself so the store can be garbage collected.
    """
    while not stop_event.wait(timeout=compaction_interval):
        if db_dir is not None and not db_dir.exists():
            stop_event.set()
            return
        if len(manifest) > max_segments_before_compact:
            logger.info("Background compaction triggered")
            compact_fn()
