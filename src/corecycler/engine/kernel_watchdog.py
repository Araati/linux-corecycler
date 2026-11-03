"""Optional faster kernel stall reporting while the tuner runs.

Lowers the RCU CPU stall timeout and the lockup watchdog threshold so an
unstable Curve Optimizer offset produces an RCU stall / soft lockup warning in
seconds instead of a minute, and restores the original values afterwards.
Both are runtime-only knobs: a reboot resets them anyway.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

RCU_STALL_TIMEOUT = Path("/sys/module/rcupdate/parameters/rcu_cpu_stall_timeout")
WATCHDOG_THRESH = Path("/proc/sys/kernel/watchdog_thresh")
# If any of these is 1, a stall/lockup PANICS the kernel instead of warning.
PANIC_KNOBS = (
    Path("/proc/sys/kernel/softlockup_panic"),
    Path("/proc/sys/kernel/hardlockup_panic"),
    Path("/proc/sys/kernel/panic_on_rcu_stall"),
)
FAST_VALUE = "5"


class FastStallDetection:
    """apply() lowers the thresholds; restore() puts back what apply() found."""

    def __init__(
        self,
        knobs: tuple[Path, ...] | None = None,
        panic_knobs: tuple[Path, ...] | None = None,
    ) -> None:
        # Module globals read at construction, so tests can redirect them.
        self._knobs = knobs if knobs is not None else (RCU_STALL_TIMEOUT, WATCHDOG_THRESH)
        self._panic_knobs = panic_knobs if panic_knobs is not None else PANIC_KNOBS
        self._originals: dict[Path, str] = {}

    @property
    def active(self) -> bool:
        return bool(self._originals)

    def apply(self) -> list[str]:
        """Lower every knob to FAST_VALUE. Returns messages for the user log."""
        messages: list[str] = []
        for knob in self._knobs:
            if knob in self._originals:
                continue
            try:
                original = knob.read_text().strip()
                knob.write_text(FAST_VALUE)
            except OSError as e:
                messages.append(f"Fast stall detection: cannot set {knob} ({e})")
                continue
            self._originals[knob] = original
            messages.append(f"Fast stall detection: {knob} {original} -> {FAST_VALUE}")
        for knob in self._panic_knobs:
            try:
                if knob.read_text().strip() not in ("", "0"):
                    messages.append(f"WARNING: {knob} is enabled — a stall will panic (reboot) the machine, not warn")
            except OSError:
                continue
        for m in messages:
            log.info(m)
        return messages

    def restore(self) -> list[str]:
        """Write back the values apply() replaced."""
        messages: list[str] = []
        for knob, original in list(self._originals.items()):
            try:
                knob.write_text(original)
                messages.append(f"Fast stall detection: restored {knob} to {original}")
            except OSError as e:
                messages.append(f"Fast stall detection: could not restore {knob} to {original} ({e})")
                log.warning("could not restore %s to %s: %s", knob, original, e)
            del self._originals[knob]
        for m in messages:
            log.info(m)
        return messages
