"""
Pipeline-bounded batching for the MANY (upload / update / delete) operations.

Design
------
* A producer thread feeds raw records into a bounded queue, so a large input
  never materialises fully in memory.
* ``workers`` consumer threads pull records, accumulate ``batch_size`` of them
  per batch and hand each batch to a ``sink`` callable (e.g. an HTTP POST).
* ``max_queue`` bounds how far the producer can run ahead of the consumers,
  giving back-pressure against runaway memory growth.

The generic engine is used by :mod:`client.many_docs2` for uploads, updates
(each ParallelRecord becomes one request row) and delete-many (each id becomes
one request row).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from queue import Empty, Queue

T = object


class BoundedPipeline:
    """Run a producer/consumer batching pipeline over a bounded queue.

    ``transform`` maps one raw item to the payload line submitted to ``sink``.
    Items are accumulated into batches of ``batch_size`` and ``sink(batch)`` is
    invoked from the consumer thread for each batch (concurrency across
    batches preserves throughput while bounding memory).

    Callers that want row-level progress can pass a ``progress`` callback; it
    receives the running total of dispatched payloads after every batch.
    """

    def __init__(self,
                 sink: Callable[[list], None],
                 transform: Callable[[object], T],
                 batch_size: int = 64,
                 workers: int = 4,
                 max_queue: int = 256,
                 queue_timeout: float = 60.0,
                 progress: Callable[[int], None] | None = None) -> None:
        self.sink = sink
        self.transform = transform
        self.batch_size = max(1, int(batch_size))
        self.workers = max(1, int(workers))
        self.max_queue = max(1, int(max_queue))
        self.queue_timeout = queue_timeout
        self.progress = progress
        self._in: Queue[object | None] = Queue(maxsize=self.max_queue)
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._sent = 0

    def feed(self, items: Iterable) -> None:
        """Stream ``items`` through the pipeline until drained and joined.

        Any sink/transform errors are recorded (not raised) so a single bad
        batch cannot kill the whole ingest; inspect :attr:`errors` afterwards.
        """
        self._errors.clear()
        self._sent = 0

        producer = threading.Thread(target=self._producer, args=(items,), daemon=True)
        consumers = [
            threading.Thread(target=self._consumer, daemon=True)
            for _ in range(self.workers)
        ]

        producer.start()
        for c in consumers:
            c.start()

        for c in consumers:
            c.join()

    def _producer(self, items: Iterable) -> None:
        try:
            for raw in items:
                self._in.put(raw, timeout=self.queue_timeout)
        except Exception as e:
            self._record(f"producer: {e}")
        finally:
            for _ in range(self.workers):
                self._in.put(None, timeout=self.queue_timeout)

    def _consumer(self) -> None:
        batch: list = []
        while True:
            if len(batch) >= self.batch_size:
                self._dispatch(batch)
                batch = []
            try:
                raw = self._in.get(timeout=self.queue_timeout)
            except Empty:
                # The producer guarantees it eventually pushes the sentinels
                # (put() has a timeout) even if the source stalls, so just
                # keep waiting rather than racing on emptiness.
                continue
            if raw is None:
                break
            try:
                batch.append(self.transform(raw))
            except Exception as e:
                self._record(f"transform: {e}")
        if batch:
            self._dispatch(batch)

    def _dispatch(self, batch: list) -> None:
        try:
            self.sink(batch)
        except Exception as e:
            self._record(f"sink: {e}")
            return
        with self._lock:
            self._sent += len(batch)
        if self.progress is not None:
            try:
                self.progress(self._sent)
            except Exception:
                pass

    def _record(self, msg: str) -> None:
        with self._lock:
            self._errors.append(msg)

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    @property
    def sent(self) -> int:
        return self._sent