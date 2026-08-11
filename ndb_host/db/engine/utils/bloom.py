"""
Probabilistic Bloom filter for fast existence checks.
Uses bytearray bits and deterministic zlib.crc32 hashing.
"""

import zlib
from typing import Any

from .config import DatabaseConfig

class BloomFilter:
    """
    Space‑efficient probabilistic structure.  Tuned via bits_per_key and
    hash_count.  Can be serialised to bytes and reconstructed.
    """
    def __init__(
        self,
        num_keys: int,
        bits_per_key: int | None = None,
        hash_count: int | None = None
    ) -> None:
        # Use config defaults if not provided
        self.bits_per_key = bits_per_key if bits_per_key is not None else DatabaseConfig.BLOOM_FILTER_BITS_PER_KEY
        self.hash_count = hash_count if hash_count is not None else DatabaseConfig.BLOOM_FILTER_HASH_COUNT
        self.size = max(1, num_keys * self.bits_per_key)
        self.bits = bytearray((self.size + 7) // 8)
        self.num_keys = 0

    def _hash(self, key_bytes: bytes, seed: int) -> int:
        """CRC32‑based deterministic hash, seeded to produce independent bits."""
        h = zlib.crc32(key_bytes + seed.to_bytes(4, 'little')) & 0xFFFFFFFF
        return h % self.size

    def add(self, key: Any) -> None:
        """Insert a key into the filter."""
        key_bytes = str(key).encode('utf-8')
        for i in range(self.hash_count):
            pos = self._hash(key_bytes, i)
            byte_idx = pos // 8
            bit_idx = pos % 8
            self.bits[byte_idx] |= (1 << bit_idx)
        self.num_keys += 1

    def might_contain(self, key: Any) -> bool:
        """Check for probable presence.  False positives possible, no false negatives."""
        key_bytes = str(key).encode('utf-8')
        for i in range(self.hash_count):
            pos = self._hash(key_bytes, i)
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self.bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def to_bytes(self) -> bytes:
        """Return the raw bit array as bytes."""
        return bytes(self.bits)

    @staticmethod
    def from_bytes(
        data: bytes,
        num_keys: int,
        bits_per_key: int | None = None,
        hash_count: int | None = None
    ) -> 'BloomFilter':
        """Reconstruct a Bloom filter from its serialised bit array."""
        bf = BloomFilter(num_keys, bits_per_key, hash_count)
        if len(data) < len(bf.bits):
            data = data + b'\x00' * (len(bf.bits) - len(data))
        bf.bits = bytearray(data[:len(bf.bits)])
        return bf
