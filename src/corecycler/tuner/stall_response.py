"""Immediate response to a kernel stall, off the tuner's main thread.

A kernel stall (RCU stall, soft/hard lockup) can be followed by a frozen
machine within seconds. The tuner's normal result path runs only after the
stress process has exited and the Qt main thread has picked the verdict up,
and either may never happen. So the moment the supervisor pins a stall on a
core, this responder restores that core's Curve Optimizer value and writes the
failure row, each on its own daemon thread so one hanging cannot block the
other or the supervisor.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import persistence as tp

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StallTarget:
    """What the responder needs for one core, captured when the test starts
    (the worker thread must not read the tuner's live state)."""

    offset: int  # the CO value under test (resident during the test)
    baseline: int  # the value to restore
    phase: str  # tuner_test_log phase label
    backend: str | None = None
    stress_mode: str | None = None
    fft_preset: str | None = None


class StallResponder:
    """Callable for SuperviseHooks.on_kernel_stall. Never raises, never blocks."""

    def __init__(
        self,
        *,
        smu,
        db_path: str | Path | None,
        session_id: int | None,
        targets: dict[int, StallTarget],
    ) -> None:
        self._smu = smu
        self._db_path = db_path
        self._session_id = session_id
        self._targets = dict(targets)
        self._lock = threading.Lock()
        self._row_threads: dict[int, threading.Thread] = {}
        self._rows: dict[int, int] = {}  # core -> row id, once committed
        self._co_threads: dict[int, threading.Thread] = {}

    def __call__(self, core_id: int, message: str, duration: float) -> None:
        try:
            self._respond(core_id, message, duration)
        except Exception:
            log.exception("kernel stall response failed for core %d", core_id)

    def _respond(self, core_id: int, message: str, duration: float) -> None:
        target = self._targets.get(core_id)
        if target is None:
            log.warning("Kernel stall on core %d: core was not captured at test start; no emergency write", core_id)
            return
        with self._lock:
            if core_id in self._co_threads:
                return  # once per test
            co = threading.Thread(
                target=self._restore_co,
                args=(core_id, target),
                name=f"stall-co-{core_id}",
                daemon=True,
            )
            self._co_threads[core_id] = co
            row = None
            if self._db_path is not None and self._session_id is not None:
                row = threading.Thread(
                    target=self._write_row,
                    args=(core_id, target, message, duration),
                    name=f"stall-row-{core_id}",
                    daemon=True,
                )
                self._row_threads[core_id] = row
        co.start()
        if row is not None:
            row.start()

    def _restore_co(self, core_id: int, target: StallTarget) -> None:
        if self._smu is None:
            return
        try:
            ok = self._smu.set_co_offset(core_id, target.baseline)
        except Exception as e:
            log.error("Kernel stall: CO restore of core %d to %d failed: %s", core_id, target.baseline, e)
            return
        if ok:
            log.info("Kernel stall: core %d CO restored %d -> %d", core_id, target.offset, target.baseline)
        else:
            log.error("Kernel stall: CO restore of core %d to %d was not verified", core_id, target.baseline)

    def _write_row(self, core_id: int, target: StallTarget, message: str, duration: float) -> None:
        try:
            row_id = tp.log_test_failure_durable(
                self._db_path,
                self._session_id,
                core_id,
                target.offset,
                target.phase,
                message,
                "kernel_stall",
                duration,
                backend=target.backend,
                stress_mode=target.stress_mode,
                fft_preset=target.fft_preset,
            )
        except Exception as e:
            log.error("Kernel stall: could not write the failure row for core %d: %s", core_id, e)
            return
        with self._lock:
            self._rows[core_id] = row_id
        log.info(
            "Kernel stall: failure of core %d at CO %d recorded (row %d, fsync'd)",
            core_id,
            target.offset,
            row_id,
        )

    def co_status(self, core_id: int, timeout: float = 0.0) -> str:
        """'done' (finished, whatever the result), 'pending' (still running
        after ``timeout``) or 'none' (never started)."""
        with self._lock:
            thread = self._co_threads.get(core_id)
        if thread is None:
            return "none"
        thread.join(timeout)
        return "pending" if thread.is_alive() else "done"

    def row_status(self, core_id: int, timeout: float = 0.0) -> str:
        """'written', 'pending' (still writing after ``timeout``), or 'none'
        (never started or failed: the caller must write the row itself)."""
        with self._lock:
            thread = self._row_threads.get(core_id)
        if thread is None:
            return "none"
        thread.join(timeout)
        with self._lock:
            if core_id in self._rows:
                return "written"
        return "pending" if thread.is_alive() else "none"
