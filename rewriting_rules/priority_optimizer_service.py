"""Process isolation and caching for priority-aware regex optimization.

This module is deliberately infrastructure-only.  The SQL rewrite rule owns
fragment discovery, predicate encoding, and AST mutation; this service owns the
potentially expensive pure-regex call and its hard timeout boundary.  Keeping
that boundary separate makes the algorithmic pipeline in
``regex_optimization_plus`` visible without changing the worker lifecycle or
cache behavior.
"""

from __future__ import annotations

import atexit
import multiprocessing
import threading
import time
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from priority_regex_optimizer import (
    EquivalenceMode,
    ProofScope,
    Verdict,
    optimize_priority_regex,
    verify_match_equivalent,
    verify_priority_equivalent,
)


class PriorityOptimizerConfig(NamedTuple):
    """Proof contract and limits that determine a cached optimizer result."""

    max_candidates: int
    output_length_factor: float
    max_nfa_states: int
    max_product_states: int
    equivalence_mode: EquivalenceMode = EquivalenceMode.LANGUAGE_AND_MATCH


_OptimizerPayload = Tuple[str, ...]


class PriorityOptimizerOutcome(NamedTuple):
    """A proved candidate together with the scope of that proof."""

    pattern: str
    proof_scope: ProofScope


def _optimize_payload(
    source: str, config: PriorityOptimizerConfig
) -> _OptimizerPayload:
    try:
        result = optimize_priority_regex(
            source,
            max_candidates=config.max_candidates,
            max_output_length_factor=config.output_length_factor,
            max_nfa_states=config.max_nfa_states,
            max_product_states=config.max_product_states,
            equivalence_mode=config.equivalence_mode,
        )
        return (
            "ok",
            result.optimized,
            result.verification.verdict.value,
            result.verification.reason,
            result.proof_scope.value,
        )
    except BaseException as exc:  # child errors always mean fail closed
        return ("error", f"{type(exc).__name__}: {exc}")


def _optimizer_worker(connection) -> None:
    """Serve optimizer batches until shutdown, retaining Greenery caches."""

    try:
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                break
            if not message or message[0] == "shutdown":
                break

            command, request_id = message[:2]
            if command == "verify":
                try:
                    _, _, original, candidate, config = message
                    mode = EquivalenceMode(config.equivalence_mode)
                    verifier = (
                        verify_match_equivalent
                        if mode == EquivalenceMode.MATCH_ONLY
                        else verify_priority_equivalent
                    )
                    verification = verifier(
                        original,
                        candidate,
                        max_nfa_states=config.max_nfa_states,
                        max_product_states=config.max_product_states,
                    )
                    connection.send(
                        ("verification", request_id, verification.verdict.value)
                    )
                except (BrokenPipeError, EOFError, OSError):
                    return
                except BaseException:
                    try:
                        connection.send(
                            ("verification", request_id, Verdict.UNKNOWN.value)
                        )
                    except (BrokenPipeError, EOFError, OSError):
                        return
                continue
            if command == "optimize_many":
                _, _, sources, config = message
                for index, source in enumerate(sources):
                    try:
                        connection.send(("started", request_id, index))
                        payload = _optimize_payload(source, config)
                        connection.send(("result", request_id, index, payload))
                    except (BrokenPipeError, EOFError, OSError):
                        return
    finally:
        connection.close()


class PriorityOptimizerService:
    """One restartable spawn worker shared by rule instances in this process."""

    MAX_CACHED_RESULTS = 2048

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connection = None
        self._process = None
        self._request_id = 0
        # Cache the complete proved result, not only Greenery's intermediate
        # reduction.  Exact product verification is deterministic for a source
        # and configuration and is often the dominant cost on repeated shapes.
        # Configuration includes the proof mode: a MATCH_ONLY result cannot
        # satisfy a later request for language-and-match equivalence.
        self._result_cache: OrderedDict[
            Tuple[str, PriorityOptimizerConfig], PriorityOptimizerOutcome
        ] = OrderedDict()
        self._verification_cache: OrderedDict[
            Tuple[str, str, PriorityOptimizerConfig], bool
        ] = OrderedDict()

    def _cached_result_locked(
        self, source: str, config: PriorityOptimizerConfig
    ) -> Optional[PriorityOptimizerOutcome]:
        key = (source, config)
        result = self._result_cache.get(key)
        if result is not None:
            self._result_cache.move_to_end(key)
        return result

    def _cache_result_locked(
        self,
        source: str,
        config: PriorityOptimizerConfig,
        result: PriorityOptimizerOutcome,
    ) -> None:
        key = (source, config)
        self._result_cache[key] = result
        self._result_cache.move_to_end(key)
        while len(self._result_cache) > self.MAX_CACHED_RESULTS:
            self._result_cache.popitem(last=False)

    def _ensure_worker_locked(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._discard_worker_locked()
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_optimizer_worker,
            args=(child_connection,),
            name="regex-priority-optimizer",
        )
        process.daemon = True
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

    def _discard_worker_locked(self, *, graceful: bool = False) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            if graceful:
                try:
                    connection.send(("shutdown",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            connection.close()
        if process is None or process.pid is None:
            return
        process.join(timeout=0.2 if graceful else 0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.5)
        if not process.is_alive():
            process.close()

    def optimize_many(
        self,
        sources: Sequence[str],
        config: PriorityOptimizerConfig,
        timeout_seconds: float,
    ) -> List[Optional[PriorityOptimizerOutcome]]:
        """Return proved results, or ``None`` for failures and timeouts."""

        if not sources or multiprocessing.current_process().daemon:
            return [None] * len(sources)

        with self._lock:
            results: List[Optional[PriorityOptimizerOutcome]] = [None] * len(sources)
            missing_sources: List[str] = []
            missing_result_indices: List[List[int]] = []
            missing_index_by_source: Dict[str, int] = {}
            for result_index, source in enumerate(sources):
                cached = self._cached_result_locked(source, config)
                if cached is not None:
                    results[result_index] = cached
                    continue
                missing_index = missing_index_by_source.get(source)
                if missing_index is None:
                    missing_index = len(missing_sources)
                    missing_index_by_source[source] = missing_index
                    missing_sources.append(source)
                    missing_result_indices.append([])
                missing_result_indices[missing_index].append(result_index)

            if not missing_sources:
                return results

            try:
                self._ensure_worker_locked()
                assert self._connection is not None
                self._request_id += 1
                request_id = self._request_id
                self._connection.send(
                    ("optimize_many", request_id, missing_sources, config)
                )
                pending = set(range(len(missing_sources)))
                deadline = time.monotonic() + timeout_seconds
                while pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._connection.poll(remaining):
                        self._discard_worker_locked()
                        break
                    message = self._connection.recv()
                    if len(message) < 2 or message[1] != request_id:
                        continue
                    if message[0] == "started":
                        continue
                    if message[0] != "result":
                        continue
                    _, _, missing_index, payload = message
                    source = missing_sources[missing_index]
                    optimized: Optional[PriorityOptimizerOutcome] = None
                    if (
                        payload
                        and payload[0] == "ok"
                        and len(payload) >= 5
                        and payload[2] == Verdict.PROVED.value
                    ):
                        try:
                            optimized = PriorityOptimizerOutcome(
                                payload[1], ProofScope(payload[4])
                            )
                        except ValueError:
                            optimized = None
                    if optimized is not None:
                        self._cache_result_locked(source, config, optimized)
                    for result_index in missing_result_indices[missing_index]:
                        results[result_index] = optimized
                    pending.discard(missing_index)
            except (BrokenPipeError, EOFError, OSError):
                self._discard_worker_locked()
            return results

    def close(self) -> None:
        with self._lock:
            self._discard_worker_locked(graceful=True)

    def verify_equivalent(
        self,
        original: str,
        candidate: str,
        config: PriorityOptimizerConfig,
        timeout_seconds: float,
    ) -> bool:
        """Prove one complete-context pair behind the hard timeout boundary."""

        if timeout_seconds <= 0 or multiprocessing.current_process().daemon:
            return False
        with self._lock:
            cache_key = (original, candidate, config)
            cached = self._verification_cache.get(cache_key)
            if cached is not None:
                self._verification_cache.move_to_end(cache_key)
                return cached
            try:
                self._ensure_worker_locked()
                assert self._connection is not None
                self._request_id += 1
                request_id = self._request_id
                self._connection.send(
                    ("verify", request_id, original, candidate, config)
                )
                if not self._connection.poll(timeout_seconds):
                    self._discard_worker_locked()
                    return False
                message = self._connection.recv()
                proved = (
                    len(message) == 3
                    and message[0] == "verification"
                    and message[1] == request_id
                    and message[2] == Verdict.PROVED.value
                )
                if (
                    len(message) == 3
                    and message[0] == "verification"
                    and message[1] == request_id
                ):
                    self._verification_cache[cache_key] = proved
                    self._verification_cache.move_to_end(cache_key)
                    while len(self._verification_cache) > self.MAX_CACHED_RESULTS:
                        self._verification_cache.popitem(last=False)
                return proved
            except (BrokenPipeError, EOFError, OSError):
                self._discard_worker_locked()
                return False


PRIORITY_OPTIMIZER_SERVICE = PriorityOptimizerService()
atexit.register(PRIORITY_OPTIMIZER_SERVICE.close)
