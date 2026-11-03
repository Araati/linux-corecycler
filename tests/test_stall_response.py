"""Immediate kernel-stall response: CO restore and an fsync'd failure row
written from the supervisor thread, before the stress process exits and
before the tuner's main thread sees the verdict."""

from __future__ import annotations

import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from corecycler.history.db import HistoryDB
from corecycler.tuner import persistence as tp
from corecycler.tuner.config import TunerConfig
from corecycler.tuner.engine import TunerEngine
from corecycler.tuner.stall_response import StallResponder, StallTarget
from corecycler.tuner.state import CoreState, TunerPhase

MSG = "Kernel stall during stress: rcu: INFO: rcu_preempt detected stalls on CPUs/tasks:"


@pytest.fixture
def file_db(tmp_path):
    d = HistoryDB(tmp_path / "history.db")
    yield d
    d.close()


def _smu():
    smu = MagicMock()
    smu.commands.co_range = (-60, 10)
    smu.set_co_offset = MagicMock(return_value=True)
    smu.get_co_offset = MagicMock(return_value=0)
    smu.get_all_co_offsets = MagicMock(return_value={0: 0, 1: 0, 2: 0, 3: 0})
    smu.get_pbo_scalar = MagicMock(return_value=1.0)
    smu.get_boost_limit = MagicMock(return_value=5500)
    return smu


def _rows_from_other_connection(path, sid):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(
            "SELECT core_id, offset_tested, phase, passed, error_type, duration_seconds, backend "
            "FROM tuner_test_log WHERE session_id=? ORDER BY id",
            (sid,),
        ).fetchall()
    finally:
        conn.close()


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _session(db):
    return tp.create_session(db, TunerConfig(cores_to_test=[0]), "", "")


TARGET = StallTarget(offset=-50, baseline=0, phase="hardening_t1", backend="mprime", stress_mode="AVX2")


class TestStallResponder:
    def test_row_is_durable_and_co_is_restored_within_a_second(self, file_db):
        sid = _session(file_db)
        smu = MagicMock()
        smu.set_co_offset.return_value = True
        responder = StallResponder(smu=smu, db_path=file_db.path, session_id=sid, targets={0: TARGET})

        started = time.monotonic()
        responder(0, MSG, 31.5)
        assert time.monotonic() - started < 0.2  # returns at once

        assert _wait_for(lambda: _rows_from_other_connection(file_db.path, sid))
        assert _rows_from_other_connection(file_db.path, sid) == [
            (0, -50, "hardening_t1", 0, "kernel_stall", 31.5, "mprime")
        ]
        assert _wait_for(lambda: smu.set_co_offset.called)
        smu.set_co_offset.assert_called_once_with(0, 0)
        assert responder.row_status(0, timeout=1.0) == "written"

    def test_a_hung_smu_write_does_not_stop_the_row(self, file_db):
        sid = _session(file_db)
        hang = threading.Event()
        smu = MagicMock()
        smu.set_co_offset.side_effect = lambda *_a: hang.wait(10)
        responder = StallResponder(smu=smu, db_path=file_db.path, session_id=sid, targets={0: TARGET})
        try:
            responder(0, MSG, 5.0)
            assert _wait_for(lambda: _rows_from_other_connection(file_db.path, sid))
        finally:
            hang.set()

    def test_a_hung_db_write_does_not_stop_the_co_restore(self, file_db, monkeypatch):
        sid = _session(file_db)
        hang = threading.Event()
        monkeypatch.setattr(tp, "log_test_failure_durable", lambda *a, **k: hang.wait(10))
        smu = MagicMock()
        smu.set_co_offset.return_value = True
        responder = StallResponder(smu=smu, db_path=file_db.path, session_id=sid, targets={0: TARGET})
        try:
            responder(0, MSG, 5.0)
            assert _wait_for(lambda: smu.set_co_offset.called)
            assert responder.row_status(0, timeout=0.1) == "pending"
        finally:
            hang.set()

    def test_a_failed_db_write_hands_the_row_back(self, tmp_path):
        responder = StallResponder(smu=None, db_path=tmp_path / "missing" / "x.db", session_id=1, targets={0: TARGET})
        responder(0, MSG, 5.0)
        assert responder.row_status(0, timeout=2.0) == "none"

    def test_responds_once_per_test(self, file_db):
        sid = _session(file_db)
        responder = StallResponder(smu=None, db_path=file_db.path, session_id=sid, targets={0: TARGET})
        responder(0, MSG, 5.0)
        responder(0, MSG, 6.0)
        responder.row_status(0, timeout=2.0)
        assert len(_rows_from_other_connection(file_db.path, sid)) == 1

    def test_unknown_core_and_exceptions_never_raise(self):
        responder = StallResponder(smu=None, db_path=None, session_id=None, targets={})
        responder(7, MSG, 1.0)  # not captured: logs, no raise
        assert responder.row_status(7) == "none"


class TestEngineWiring:
    def _engine(self, db, topo, smu, backend):
        cfg = TunerConfig(cores_to_test=[0], coarse_step=5, fine_step=1, max_offset=-60)
        eng = TunerEngine(db=db, topology=topo, smu=smu, backend=backend, config=cfg)
        sid = tp.create_session(db, cfg, "", "")
        eng._session_id = sid
        cs = CoreState(core_id=0, phase=TunerPhase.COARSE_SEARCH, current_offset=-20, best_offset=-15, in_test=True)
        eng._core_states = {0: cs}
        eng._co_applied = {0: -20}
        tp.save_core_state(db, sid, cs)
        return eng, sid

    def test_stall_writes_one_row_restores_co_and_the_normal_path_adds_none(
        self, file_db, topo_single_ccd, mock_backend
    ):
        smu = _smu()
        eng, sid = self._engine(file_db, topo_single_ccd, smu, mock_backend)
        runner = MagicMock()
        eng._attach_stall_responder(runner, [0], ("mprime", "SSE", "SMALL"))

        # The supervisor thread calls this at detection time.
        runner.on_kernel_stall(0, MSG, 12.0)
        assert _wait_for(lambda: _rows_from_other_connection(file_db.path, sid))
        assert _wait_for(lambda: smu.set_co_offset.called)
        assert smu.set_co_offset.call_args.args == (0, 0)  # baseline

        # Later, if the machine survives, the queued verdict reaches the main thread.
        with patch("corecycler.tuner.engine.QTimer"):
            eng._on_test_finished(0, False, MSG, "kernel_stall", 12.0, 0.0)
        rows = _rows_from_other_connection(file_db.path, sid)
        assert [(r[0], r[1], r[2], r[4]) for r in rows] == [(0, -20, "coarse", "kernel_stall")]

    def test_normal_path_writes_the_row_when_the_emergency_write_failed(
        self, file_db, topo_single_ccd, mock_backend, monkeypatch
    ):
        eng, sid = self._engine(file_db, topo_single_ccd, _smu(), mock_backend)

        def fail(*_a, **_k):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(tp, "log_test_failure_durable", fail)
        runner = MagicMock()
        eng._attach_stall_responder(runner, [0], ("mprime", "SSE", "SMALL"))
        runner.on_kernel_stall(0, MSG, 12.0)
        with patch("corecycler.tuner.engine.QTimer"):
            eng._on_test_finished(0, False, MSG, "kernel_stall", 12.0, 0.0)
        assert [r[4] for r in _rows_from_other_connection(file_db.path, sid)] == ["kernel_stall"]

    def test_scheduler_supervisor_receives_the_hook(self, topo_single_ccd, mock_backend):
        from corecycler.engine.backends.base import StressConfig
        from corecycler.engine.scheduler import CoreScheduler, SchedulerConfig

        sched = CoreScheduler(
            topology=topo_single_ccd,
            backend=mock_backend,
            stress_config=StressConfig(),
            scheduler_config=SchedulerConfig(cores_to_test=[0]),
        )
        hook = MagicMock()
        sched.on_kernel_stall = hook
        assert sched._supervisor("stress").hooks.on_kernel_stall is hook


class TestCoRestoreOrdering:
    def test_unfinished_emergency_restore_pauses_instead_of_advancing(self, file_db, topo_single_ccd, mock_backend):
        hang = threading.Event()
        smu = _smu()
        eng, sid = TestEngineWiring()._engine(file_db, topo_single_ccd, smu, mock_backend)
        runner = MagicMock()
        eng._attach_stall_responder(runner, [0], ("mprime", "SSE", "SMALL"))
        smu.set_co_offset.side_effect = lambda *_a: hang.wait(30)  # SMU write stuck
        runner.on_kernel_stall(0, MSG, 12.0)
        try:
            with (
                patch("corecycler.tuner.engine.QTimer"),
                patch.object(eng, "_advance_core") as advance,
            ):
                eng._on_test_finished(0, False, MSG, "kernel_stall", 12.0, 0.0)
            advance.assert_not_called()
            assert eng._paused
            # the only SMU call is the emergency restore: no back-off write raced it
            assert smu.set_co_offset.call_count == 1
        finally:
            hang.set()

    def test_co_status_reports_pending_then_done(self):
        hang = threading.Event()
        smu = MagicMock()
        smu.set_co_offset.side_effect = lambda *_a: hang.wait(10)
        responder = StallResponder(smu=smu, db_path=None, session_id=None, targets={0: TARGET})
        responder(0, MSG, 1.0)
        assert responder.co_status(0, timeout=0.05) == "pending"
        hang.set()
        assert responder.co_status(0, timeout=2.0) == "done"
        assert responder.co_status(5) == "none"


class TestResumeAfterFreeze:
    """The 2026-11-03 state: the emergency row landed, then the box froze with
    core 0 still in_test. Resume's crash recovery must handle it once."""

    def test_one_crash_penalty_and_no_apparatus_trip(self, file_db, topo_single_ccd, mock_backend):
        cfg = TunerConfig(cores_to_test=[0], coarse_step=5, fine_step=1, max_offset=-60, crash_penalty_steps=3)
        eng = TunerEngine(db=file_db, topology=topo_single_ccd, smu=_smu(), backend=mock_backend, config=cfg)
        sid = tp.create_session(file_db, cfg, "", "")
        eng._session_id = sid
        tp.update_session_status(file_db, sid, "running")
        cs = CoreState(
            core_id=0,
            phase=TunerPhase.HARDENING_T1,
            current_offset=-50,
            best_offset=-50,
            baseline_offset=0,
            in_test=True,
        )
        eng._core_states = {0: cs}
        tp.save_core_state(file_db, sid, cs)
        tp.log_test_failure_durable(file_db.path, sid, 0, -50, "hardening_t1", MSG, "kernel_stall", 80.0)

        messages = []
        eng.log_message.connect(messages.append)
        with patch.object(eng, "_forensics", return_value=([], True)):
            crashed, _ = eng._attribute_crash_after_reboot(tp.get_session(file_db, sid))

        assert crashed == [0]
        assert cs.crash_count == 1
        assert cs.in_test is False
        assert not any("APPARATUS SUSPECT" in m for m in messages)
