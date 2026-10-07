"""Per-frame memoization keyed by object identity — a drop-in replacement for caching on
`DataFrame.attrs`.

Why this exists: pandas deep-copies `df.attrs` on *every* derived operation (its internal
`__finalize__` runs on each slice, column access, and arithmetic op). Caching a large object
on `.attrs` therefore makes every one of the millions of pandas ops in a scan O(payload) —
profiling showed ~94% of scan time spent in `copy.deepcopy` of these attrs payloads.

This caches by `id(frame)` instead. A weakref callback evicts the entry when the frame is
garbage-collected, so:
  * the cache "travels with" the exact frame object (not its derived slices), and
  * id() reuse can't cause a stale hit — the entry is removed before the dead frame's id can
    be handed to a new object (weakref callbacks run during collection).
DataFrames are unhashable but weak-referenceable, which is why id()+weakref is used rather than
a WeakKeyDictionary.

Lifetime matches the old `.attrs` behavior exactly: a transient per-scan slice caches for the
duration of that scan and is evicted when the slice is dropped; a long-lived frame caches until
it is replaced.
"""
from __future__ import annotations

import weakref
from typing import Any


class FrameCache:
    def __init__(self) -> None:
        self._data: dict[int, Any] = {}     # id(frame) -> cached value
        self._refs: dict[int, weakref.ref] = {}   # id(frame) -> weakref (keeps the evict callback alive)

    def get(self, frame, default=None):
        return self._data.get(id(frame), default)

    def put(self, frame, value):
        k = id(frame)
        if k not in self._refs:
            data, refs = self._data, self._refs

            def _evict(_ref, k=k, data=data, refs=refs):
                data.pop(k, None)
                refs.pop(k, None)

            try:
                self._refs[k] = weakref.ref(frame, _evict)
            except TypeError:
                return value     # not weak-referenceable -> skip caching rather than risk a leak
        self._data[k] = value
        return value

    def clear(self) -> None:
        self._data.clear()
        self._refs.clear()
