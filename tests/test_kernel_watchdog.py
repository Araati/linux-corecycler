"""Fast kernel stall detection: lower the thresholds while tuning, restore after."""

from __future__ import annotations

import sys as _sys

import pytest

from corecycler.engine.kernel_watchdog import FastStallDetection


def _knobs(tmp_path):
    rcu, wd = tmp_path / "rcu", tmp_path / "wd"
    rcu.write_text("60\n")
    wd.write_text("10\n")
    return rcu, wd


class TestFastStallDetection:
    def test_apply_then_restore_round_trips(self, tmp_path):
        rcu, wd = _knobs(tmp_path)
        fast = FastStallDetection(knobs=(rcu, wd), panic_knobs=())
        fast.apply()
        assert (rcu.read_text(), wd.read_text()) == ("5", "5")
        assert fast.active
        fast.apply()  # idempotent: the original is not overwritten by "5"
        fast.restore()
        assert (rcu.read_text(), wd.read_text()) == ("60", "10")
        assert not fast.active

    def test_unwritable_knob_is_reported_and_skipped(self, tmp_path):
        rcu, _ = _knobs(tmp_path)
        fast = FastStallDetection(knobs=(rcu, tmp_path / "missing"), panic_knobs=())
        messages = fast.apply()
        assert rcu.read_text() == "5"
        assert any("cannot set" in m for m in messages)
        fast.restore()
        assert rcu.read_text() == "60"

    def test_panic_knob_warning(self, tmp_path):
        rcu, _ = _knobs(tmp_path)
        panic = tmp_path / "softlockup_panic"
        panic.write_text("1\n")
        messages = FastStallDetection(knobs=(rcu,), panic_knobs=(panic,)).apply()
        assert any("panic" in m for m in messages)


@pytest.mark.skipif(not hasattr(_sys.modules.get("PySide6", None), "__path__"), reason="needs PySide6")
class TestTunerTabLifecycle:
    def test_checked_box_lowers_while_running_and_restores_on_pause(self, fake_kernel_watchdog_knobs):
        from PySide6.QtWidgets import QApplication

        from corecycler.gui.tuner_tab import TunerTab

        assert QApplication.instance() or QApplication([])
        rcu = fake_kernel_watchdog_knobs / "rcu_cpu_stall_timeout"
        tab = TunerTab(db=None, topology=None, smu=None)
        tab._set_running_state(True)
        assert rcu.read_text() == "60"  # unchecked: untouched
        tab._set_running_state(False)
        tab.set_fast_stall_detection(True)
        tab._set_running_state(True)
        assert rcu.read_text() == "5"
        tab._on_status_changed("paused")
        assert rcu.read_text() == "60"
