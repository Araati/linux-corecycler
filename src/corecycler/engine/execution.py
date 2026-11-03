"""The one supervised stress-execution loop every test path runs through."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from corecycler.engine import containment
from corecycler.engine.backends.base import KILLED_BY_US_CODES, StressResult

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

    from corecycler.engine.backends.base import StressBackend, StressConfig
    from corecycler.engine.detector import ErrorDetector, MCEEvent

log = logging.getLogger(__name__)

STALL_GRACE_SECONDS = 5.0
ERROR_POLL_INTERVAL = 5.0
WATCHDOG_INTERVAL = 2.0
STARTUP_WINDOW_SECONDS = 2.0
CONTAINMENT_GRACE_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class Lane:
    core_id: int
    cpus: tuple[int, ...]
    work_dir: Path

    @property
    def cpu_list(self) -> str:
        return containment.cpu_list(self.cpus)


@dataclass(slots=True)
class _LaneRun:
    lane: Lane
    proc: subprocess.Popen | None = None
    verdict: StressResult | None = None
    started_at: float = 0.0
    last_active: float = 0.0
    last_watchdog: float = 0.0
    prev_times: dict[int, tuple[int, int]] = field(default_factory=dict)
    unit: str | None = None
    cgroup: str | None = None
    stdout: str = ""
    stderr: str = ""
    drained: bool = False

    @property
    def running(self) -> bool:
        return self.verdict is None and self.proc is not None and self.proc.poll() is None


@dataclass(slots=True)
class SuperviseHooks:
    on_status: Callable[[int, float], None] | None = None
    on_stall: Callable[[int], None] | None = None
    on_thermal: Callable[[float], None] | None = None
    # Called on the supervisor thread the moment a kernel stall is pinned on a
    # lane: (core_id, message, seconds into the test). It must return at once;
    # the machine may freeze seconds later, so nothing waits for the test to end.
    on_kernel_stall: Callable[[int, str, float], None] | None = None


# Highest max-temperature limit any caller may configure. Values above it are
# clamped so a typo in a profile or tuner config cannot disable thermal safety.
MAX_TEMPERATURE_LIMIT_C = 105.0


class ThermalWatch:
    """Debounced soft limit, instant hard ceiling, hysteresis after a trip."""

    HYSTERESIS = 5.0

    def __init__(
        self,
        *,
        max_temperature: float,
        grace_seconds: float,
        hard_margin: float,
        require_sensor: bool,
        read: Callable[[], float | None] | None = None,
    ) -> None:
        self.max_temperature = min(max_temperature, MAX_TEMPERATURE_LIMIT_C)
        self.grace_seconds = grace_seconds
        self.hard_margin = hard_margin
        self.require_sensor = require_sensor
        self._read = read or read_cpu_temperature
        self.tripped = False
        self._over_since: float | None = None
        self.last_temperature: float | None = None

    def safe(self) -> bool:
        temp = self._read()
        self.last_temperature = temp
        if temp is None:
            return not self.require_sensor
        limit = self.max_temperature
        if temp >= limit:
            if temp >= limit + self.hard_margin:
                self.tripped = True
                return False
            if self.tripped:
                return False
            now = time.monotonic()
            if self._over_since is None:
                self._over_since = now
            if now - self._over_since >= self.grace_seconds:
                self.tripped = True
                return False
            return True
        self._over_since = None
        if self.tripped:
            if temp < limit - self.HYSTERESIS:
                self.tripped = False
                return True
            return False
        return True


def read_cpu_temperature() -> float | None:
    hwmon_base = Path("/sys/class/hwmon")
    if not hwmon_base.exists():
        return None
    with contextlib.suppress(OSError):
        for hwmon_dir in hwmon_base.iterdir():
            name_file = hwmon_dir / "name"
            if not name_file.exists():
                continue
            try:
                name = name_file.read_text().strip()
            except OSError:
                continue
            if name not in ("k10temp", "coretemp", "zenpower", "zenpower3"):
                continue
            max_temp = 0.0
            for temp_input in sorted(hwmon_dir.glob("temp*_input")):
                try:
                    temp_c = int(temp_input.read_text().strip()) / 1000.0
                except (ValueError, OSError):
                    continue
                max_temp = max(max_temp, temp_c)
            if max_temp > 0:
                return max_temp
    return None


def make_preexec():
    def _preexec():
        os.setsid()
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

    return _preexec


def sigkill_process_group(proc: subprocess.Popen) -> None:
    """Send SIGKILL to the stress process group. Never waits."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def kill_process_group(proc: subprocess.Popen, *, term_grace: float = 3.0) -> None:
    """SIGTERM, wait ``term_grace`` seconds, then SIGKILL. ``term_grace=0``
    skips SIGTERM (a stalled run: a thread on a stuck CPU cannot exit)."""
    if proc.poll() is None:
        try:
            pgid = os.getpgid(proc.pid)
        except (OSError, ProcessLookupError):
            pgid = None
        if pgid is not None:
            exited = False
            if term_grace > 0:
                with contextlib.suppress(OSError, ProcessLookupError):
                    os.killpg(pgid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=term_grace)
                    exited = True
            if not exited:
                with contextlib.suppress(OSError, ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2)
        else:
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
    for stream in (proc.stdout, proc.stderr):
        if stream:
            with contextlib.suppress(OSError):
                stream.close()


def reap_zombies() -> None:
    with contextlib.suppress(ChildProcessError):
        while True:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break


def cpu_times(cpu_id: int) -> tuple[int, int] | None:
    with contextlib.suppress(OSError, ValueError, IndexError), open("/proc/stat") as f:
        prefix = f"cpu{cpu_id} "
        for line in f:
            if line.startswith(prefix):
                vals = [int(x) for x in line.split()[1:]]
                return vals[3] + vals[4], sum(vals)
    return None


def busy_fraction(prev: tuple[int, int] | None, now: tuple[int, int] | None) -> float | None:
    if prev is None or now is None:
        return None
    d_total = now[1] - prev[1]
    if d_total <= 0:
        return None
    return 1.0 - ((now[0] - prev[0]) / d_total)


class Supervisor:
    """Runs one batch of contained lanes to per-lane honest verdicts.

    A lane with no earned verdict when the batch stops early stays None:
    an invented pass would enter the evidence record as a proven offset.
    """

    def __init__(
        self,
        *,
        backend: StressBackend,
        detector: ErrorDetector,
        thermal: ThermalWatch,
        stop_event: threading.Event,
        observed: list[MCEEvent],
        poll_interval: float = 1.0,
        stall_timeout: float = 30.0,
        stop_on_first_failure: bool = True,
        phase: str = "stress",
        hooks: SuperviseHooks | None = None,
        containment_for: Callable[[tuple[int, ...]], containment.Containment | None] | None = None,
    ) -> None:
        self.backend = backend
        self.detector = detector
        self.thermal = thermal
        self.stop_event = stop_event
        self.observed = observed
        self.poll_interval = poll_interval
        self.stall_timeout = stall_timeout
        self.stop_on_first_failure = stop_on_first_failure
        self.phase = phase
        self.hooks = hooks or SuperviseHooks()
        self._containment_for = containment_for or containment.contain
        self._we_killed = False

    def run(
        self,
        lanes: list[Lane],
        config_for: Callable[[Lane], StressConfig],
        duration: float,
    ) -> dict[int, StressResult | None]:
        runs = [_LaneRun(lane=lane) for lane in lanes]
        start = time.monotonic()
        self._we_killed = False
        try:
            for run in runs:
                if not self._launch(run, config_for(run.lane), start):
                    break
            if any(run.running for run in runs):
                self._poll_until_done(runs, start, duration)
        finally:
            self._finish(runs, start, duration)
        return {run.lane.core_id: run.verdict for run in runs}

    def _launch(self, run: _LaneRun, cfg: StressConfig, batch_start: float) -> bool:
        lane = run.lane
        try:
            self.backend.prepare(lane.work_dir, cfg)
            self.backend.assert_prepared(lane.work_dir)
            contained = self._containment_for(lane.cpus)
            prefix = contained.prefix if contained is not None else []
            run.unit = contained.unit if contained is not None else None
            cmd = prefix + self.backend.get_command(cfg, lane.work_dir)
        except (OSError, RuntimeError) as exc:
            log.error("core %d: refusing stress launch: %s", lane.core_id, exc)
            self._fail(run, f"Failed to start stress test: {exc}", batch_start, error_type="startup")
            return False
        try:
            run.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(lane.work_dir),
                preexec_fn=make_preexec(),
            )
        except (OSError, RuntimeError, TypeError) as exc:
            log.error("core %d: stress process failed to start: %s", lane.core_id, exc)
            self._fail(run, f"Failed to start stress test: {exc}", batch_start, error_type="startup")
            return False
        run.started_at = time.monotonic()
        run.last_active = run.started_at
        return True

    def _poll_until_done(self, runs: list[_LaneRun], start: float, duration: float) -> None:
        deadline = start + duration
        last_error_poll = start
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            if self._poll_exits_stalls_watchdog(runs, start):
                break
            if not any(run.running for run in runs):
                break
            if not self.thermal.safe():
                temp = self.thermal.last_temperature
                if self.hooks.on_thermal and temp is not None:
                    self.hooks.on_thermal(temp)
                first = min((r for r in runs if r.verdict is None), default=None, key=lambda r: r.lane.core_id)
                if first is not None:
                    self._fail(
                        first,
                        f"CPU temperature exceeded {self.thermal.max_temperature} C safety limit during {self.phase}",
                        start,
                    )
                self.stop_event.set()
                break
            if self._apply_mce_events(runs, start):
                break
            now = time.monotonic()
            if now - last_error_poll >= ERROR_POLL_INTERVAL:
                last_error_poll = now
                if self._poll_backend_errors(runs, start):
                    break
            for run in runs:
                if run.running and self.hooks.on_status:
                    self.hooks.on_status(run.lane.core_id, now - start)
            self.stop_event.wait(self.poll_interval)

    def _apply_mce_events(self, runs: list[_LaneRun], start: float) -> bool:
        events = self.detector.check_mce()
        if not events:
            return False
        stalls = [e for e in events if e.kind == "kernel_stall"]
        events = [e for e in events if e.kind != "kernel_stall"]
        hit = self._apply_kernel_stalls(runs, stalls, start, eligible=lambda r: r.verdict is None)
        # Stall lines stay out of the observed MCE evidence: an RCU stall can
        # name a CPU that was only waiting, and observed events demote the
        # cores they name.
        self.observed.extend(events)
        cpu_to_run = {cpu: run for run in runs for cpu in run.lane.cpus}
        for event in events:
            if event.cpu == -1:
                anchor = min((r for r in runs if r.verdict is None), default=None, key=lambda r: r.lane.core_id)
                if anchor is not None:
                    anchor.verdict = StressResult(
                        core_id=anchor.lane.core_id,
                        passed=False,
                        duration_seconds=time.monotonic() - start,
                        error_message=(f"Machine check without core attribution during {self.phase}: {event.message}"),
                        error_type="mce_unattributed",
                    )
                    self.stop_event.set()
                    hit = True
                continue
            run = cpu_to_run.get(event.cpu)
            if run is not None and run.verdict is None:
                self._fail(run, f"MCE during {self.phase}: {event.message}", start, error_type="mce")
                hit = True
        return hit

    def _apply_kernel_stalls(
        self,
        runs: list[_LaneRun],
        stalls: list[MCEEvent],
        start: float,
        *,
        eligible: Callable[[_LaneRun], bool],
    ) -> bool:
        """Fail the lane under test on an RCU stall / soft lockup and stop the run.

        One lane (the per-core search): that lane fails whatever CPU the
        kernel named — it is the only core away from baseline. Several lanes
        (validation): a named CPU inside a lane fails that lane; a stall that
        names no lane's CPU is recorded as unattributed (retried without a
        verdict), never pinned on a guessed core.
        """
        if not stalls:
            return False
        live = [r for r in runs if eligible(r)]
        if not live:
            return False
        cpu_to_run = {cpu: r for r in live for cpu in r.lane.cpus}
        targets: dict[int, tuple[_LaneRun, MCEEvent]] = {}
        for event in stalls:
            run = cpu_to_run.get(event.cpu)
            if run is None and len(runs) == 1:
                run = live[0]
            if run is not None:
                targets.setdefault(run.lane.core_id, (run, event))
        for run, event in targets.values():
            run.verdict = StressResult(
                core_id=run.lane.core_id,
                passed=False,
                duration_seconds=time.monotonic() - start,
                error_message=f"Kernel stall during {self.phase}: {event.message}",
                error_type="kernel_stall",
            )
        if not targets:
            anchor = min(live, key=lambda r: r.lane.core_id)
            anchor.verdict = StressResult(
                core_id=anchor.lane.core_id,
                passed=False,
                duration_seconds=time.monotonic() - start,
                error_message=f"Kernel stall without core attribution during {self.phase}: {stalls[0].message}",
                error_type="mce_unattributed",
            )
        # Always stop now, whatever stop_on_first_failure says: the stress
        # process must be killed before the stall turns into a freeze.
        self.stop_event.set()
        self._respond_to_kernel_stall(runs, [(run, event) for run, event in targets.values()], start)
        return True

    def _respond_to_kernel_stall(
        self,
        runs: list[_LaneRun],
        targets: list[tuple[_LaneRun, MCEEvent]],
        start: float,
    ) -> None:
        """Act on a stall now, from this thread: SIGKILL every lane (no
        waiting), then hand each stalled core to the on_kernel_stall hook.

        A stall can turn into a frozen machine within seconds. The normal end
        of a run waits on the stress process (which cannot exit while one of
        its threads sits on a stuck CPU) and then on the tuner's main thread,
        so the response cannot wait for either.
        """
        for run in runs:
            if run.proc is not None and run.proc.poll() is None:
                self._we_killed = True
                sigkill_process_group(run.proc)
        elapsed = time.monotonic() - start
        # The hook runs before anything is logged: a log write can block on a
        # disk that the stall is freezing, and logging holds a lock.
        for run, _event in targets:
            if self.hooks.on_kernel_stall is None:
                continue
            try:
                self.hooks.on_kernel_stall(run.lane.core_id, run.verdict.error_message, elapsed)
            except Exception:
                log.exception("kernel stall hook failed for core %d", run.lane.core_id)
        for run, _event in targets:
            log.info(
                "Kernel stall on core %d (%s) %.1fs into the test: stress killed; %s",
                run.lane.core_id,
                self.phase,
                elapsed,
                "emergency CO restore and failure row started" if self.hooks.on_kernel_stall else "no stall hook",
            )

    def _poll_backend_errors(self, runs: list[_LaneRun], start: float) -> bool:
        for run in runs:
            if run.verdict is not None:
                continue
            err = self.backend.poll_errors(run.lane.work_dir)
            if err:
                self._fail(run, err, start)
                return self.stop_on_first_failure
        return False

    def _poll_exits_stalls_watchdog(self, runs: list[_LaneRun], start: float) -> bool:
        now = time.monotonic()
        for run in runs:
            if run.verdict is not None or run.proc is None:
                continue
            rc = run.proc.poll()
            if rc is not None:
                self._drain(run)
                if not self._we_killed and rc in KILLED_BY_US_CODES:
                    self._fail(
                        run,
                        f"Stress process killed externally (code {rc}) — possible OOM or system issue",
                        start,
                        error_type="killed",
                    )
                    return self.stop_on_first_failure
                if rc not in KILLED_BY_US_CODES and now - run.started_at < STARTUP_WINDOW_SECONDS:
                    log.warning(
                        "Stress process for core %d exited in <%.0fs (code %d) — "
                        "binary may be missing or misconfigured",
                        run.lane.core_id,
                        STARTUP_WINDOW_SECONDS,
                        rc,
                    )
                    self._fail(
                        run,
                        self._startup_message(
                            run, rc, f"stress exited at startup (code {rc}) with no work done — verdict unavailable"
                        ),
                        start,
                        error_type="startup",
                    )
                    return True
                passed, msg = self.backend.parse_output(run.stdout, run.stderr, rc)
                if passed:
                    run.verdict = StressResult(
                        core_id=run.lane.core_id,
                        passed=True,
                        duration_seconds=now - start,
                    )
                    continue
                self._fail(run, msg or f"stress exited with code {rc}", start)
                return self.stop_on_first_failure
            if run.unit is not None and now - run.last_watchdog >= WATCHDOG_INTERVAL:
                run.last_watchdog = now
                fault = self._containment_fault(run, now)
                if fault:
                    self._fail(run, fault, start, error_type="startup")
                    return True
            if now - start >= STALL_GRACE_SECONDS and self._is_stalled(run, now):
                if self.hooks.on_stall:
                    self.hooks.on_stall(run.lane.core_id)
                self._fail(
                    run,
                    f"Stress test stalled on core {run.lane.core_id} "
                    f"(CPU usage near 0 on CPUs {run.lane.cpu_list} for "
                    f"{self.stall_timeout:.0f}s)",
                    start,
                    error_type="stall",
                )
                return self.stop_on_first_failure
        return False

    def _containment_fault(self, run: _LaneRun, now: float) -> str | None:
        if run.proc is None or run.unit is None:
            return None
        if run.cgroup is None:
            run.cgroup = containment.payload_cgroup(run.proc.pid, run.unit)
        if run.cgroup is None:
            if now - run.started_at > CONTAINMENT_GRACE_SECONDS:
                return (
                    f"scope {run.unit} never adopted the stress payload within "
                    f"{CONTAINMENT_GRACE_SECONDS:.0f}s — containment fault, not a core verdict"
                )
            return None
        effective = containment.scope_effective_cpus(run.cgroup)
        if effective is None:
            return (
                f"the kernel record for scope {run.unit} vanished while the payload ran "
                "— containment fault, not a core verdict"
            )
        if effective != set(run.lane.cpus):
            return (
                f"scope {run.unit} runs on CPUs {containment.cpu_list(effective)} "
                f"instead of {run.lane.cpu_list} — containment fault, not a core verdict"
            )
        return None

    def _is_stalled(self, run: _LaneRun, now: float) -> bool:
        active = False
        any_sample = False
        for cpu in run.lane.cpus:
            cur = cpu_times(cpu)
            busy = busy_fraction(run.prev_times.get(cpu), cur)
            if cur is not None:
                run.prev_times[cpu] = cur
            if busy is not None:
                any_sample = True
                if busy > 0.05:
                    active = True
        if active or not any_sample:
            run.last_active = now
            return False
        return now - run.last_active > self.stall_timeout

    def _startup_message(self, run: _LaneRun, rc: int, fallback: str) -> str:
        """A backend that names the environment fault in its output outranks the generic line."""
        passed, msg = self.backend.parse_output(run.stdout or "", run.stderr or "", rc)
        if not passed and msg and classify_error(msg) == "startup":
            return msg
        return fallback

    def _drain(self, run: _LaneRun) -> None:
        if run.drained or run.proc is None:
            return
        try:
            run.stdout, run.stderr = run.proc.communicate(timeout=2)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            run.stdout, run.stderr = run.stdout or "", run.stderr or ""
        run.drained = True

    def _fail(
        self,
        run: _LaneRun,
        message: str,
        start: float,
        *,
        error_type: str | None = None,
    ) -> None:
        run.verdict = StressResult(
            core_id=run.lane.core_id,
            passed=False,
            duration_seconds=time.monotonic() - start,
            error_message=message,
            error_type=error_type or classify_error(message),
        )
        if self.stop_on_first_failure:
            self.stop_event.set()

    def _finish(self, runs: list[_LaneRun], start: float, duration: float) -> None:
        for run in runs:
            if run.proc is not None and run.proc.poll() is None:
                self._we_killed = True
        for run in runs:
            if run.proc is None:
                continue
            if run.verdict is not None and run.verdict.error_type == "kernel_stall":
                # Already SIGKILLed at detection. Its output is not needed for
                # the verdict, and waiting on it only delays the tuner.
                kill_process_group(run.proc, term_grace=0)
                continue
            self._drain(run)
            kill_process_group(run.proc)
        drained = self.detector.check_mce()
        stalls = [e for e in drained if e.kind == "kernel_stall"]
        drained = [e for e in drained if e.kind != "kernel_stall"]
        self._apply_kernel_stalls(runs, stalls, start, eligible=lambda r: r.verdict is None or r.verdict.passed)
        if drained:
            self.observed.extend(drained)
            cpu_to_run = {cpu: r for r in runs for cpu in r.lane.cpus}
            for event in drained:
                target = None
                if event.cpu == -1:
                    target = min(
                        (r for r in runs if r.verdict is None or r.verdict.passed),
                        default=None,
                        key=lambda r: r.lane.core_id,
                    )
                else:
                    target = cpu_to_run.get(event.cpu)
                if target is not None and (target.verdict is None or target.verdict.passed):
                    target.verdict = StressResult(
                        core_id=target.lane.core_id,
                        passed=False,
                        duration_seconds=time.monotonic() - start,
                        error_message=f"MCE during {self.phase}: {event.message}",
                        error_type="mce" if event.cpu != -1 else "mce_unattributed",
                    )
        elapsed = time.monotonic() - start
        interrupted = self.stop_event.is_set() and elapsed < duration
        for run in runs:
            if run.verdict is None and run.proc is not None:
                run.verdict = self._final_verdict(run, elapsed, interrupted)
        reap_zombies()

    def _final_verdict(self, run: _LaneRun, elapsed: float, interrupted: bool) -> StressResult | None:
        rc = run.proc.returncode if run.proc is not None else 0
        rc = rc if rc is not None else 0
        if rc in KILLED_BY_US_CODES and not self._we_killed:
            return StressResult(
                core_id=run.lane.core_id,
                passed=False,
                duration_seconds=elapsed,
                error_message=(f"Stress process killed externally (code {rc}) — possible OOM or system issue"),
                error_type="killed",
            )
        if rc != 0 and rc not in KILLED_BY_US_CODES and elapsed < STARTUP_WINDOW_SECONDS:
            return StressResult(
                core_id=run.lane.core_id,
                passed=False,
                duration_seconds=elapsed,
                error_message=self._startup_message(run, rc, f"stress exited with code {rc} at startup"),
                error_type="startup",
            )
        live_err = self.backend.poll_errors(run.lane.work_dir)
        if live_err:
            return StressResult(
                core_id=run.lane.core_id,
                passed=False,
                duration_seconds=elapsed,
                error_message=live_err,
                error_type=classify_error(live_err),
            )
        passed, msg = self.backend.parse_output(run.stdout or "", run.stderr or "", rc)
        if not passed:
            return StressResult(
                core_id=run.lane.core_id,
                passed=False,
                duration_seconds=elapsed,
                error_message=msg,
                error_type=classify_error(msg) if msg else None,
            )
        if interrupted:
            return None
        return StressResult(core_id=run.lane.core_id, passed=True, duration_seconds=elapsed)


def watch_idle(
    *,
    cpus: tuple[int, ...],
    duration: float,
    thermal: ThermalWatch,
    detector: ErrorDetector,
    stop_event: threading.Event,
    observed: list[MCEEvent],
    phase: str,
    poll_interval: float = 0.5,
    core_id: int | None = None,
    on_kernel_stall: Callable[[int, str, float], None] | None = None,
) -> str | None:
    """Watch an idle segment of the core under test. A kernel stall fails it
    and, like the supervisor, triggers ``on_kernel_stall(core_id, ...)`` at once
    (nothing runs to kill, but the tested CO value is still applied)."""
    own = set(cpus)
    start = time.monotonic()
    while time.monotonic() - start < duration and not stop_event.is_set():
        if not thermal.safe():
            stop_event.set()
            return f"CPU temperature exceeded {thermal.max_temperature} C safety limit during {phase}"
        events = detector.check_mce()
        if events:
            for event in events:
                if event.kind == "kernel_stall":
                    # Idle phase of the core under test: any stall counts against it.
                    message = f"Kernel stall during {phase}: {event.message}"
                    if on_kernel_stall is not None and core_id is not None:
                        try:
                            on_kernel_stall(core_id, message, time.monotonic() - start)
                        except Exception:
                            log.exception("kernel stall hook failed for core %d", core_id)
                    log.info("Kernel stall on core %s during %s", core_id, phase)
                    return message
            observed.extend(events)
            for event in events:
                if event.cpu == -1 or event.cpu in own:
                    return f"MCE during {phase}: {event.message}"
        remaining = duration - (time.monotonic() - start)
        stop_event.wait(min(poll_interval, max(0.0, remaining)))
    return None


def classify_error(msg: str | None) -> str:
    if not msg:
        return "unknown"
    msg_lower = msg.lower()
    if (
        "failed to start" in msg_lower
        or "verdict unavailable" in msg_lower
        or "harness error" in msg_lower
        or "containment fault" in msg_lower
    ):
        return "startup"
    if "machine check without core attribution" in msg_lower or "kernel stall without core attribution" in msg_lower:
        return "mce_unattributed"
    # Before the generic "stall" rule: a kernel stall is a stability failure,
    # the apparatus "stall" (no progress from the stress process) is not.
    if "kernel stall" in msg_lower:
        return "kernel_stall"
    if "mce" in msg_lower or "machine check" in msg_lower:
        return "mce"
    if "temperature" in msg_lower or "thermal" in msg_lower:
        return "thermal"
    if "stall" in msg_lower:
        return "stall"
    if any(
        w in msg_lower
        for w in (
            "rounding",
            "fatal",
            "illegal",
            "sumout",
            "mismatch",
            "jacobi",
            "verification",
            "computation",
        )
    ):
        return "computation"
    if "timeout" in msg_lower:
        return "timeout"
    if "killed externally" in msg_lower:
        return "killed"
    if "crash" in msg_lower or "signal" in msg_lower:
        return "crash"
    if re.search(r"exited with code -\d+", msg_lower):
        return "crash"
    if "idle" in msg_lower:
        return "idle_instability"
    if "variable" in msg_lower or "transition" in msg_lower:
        return "load_transition"
    return "unknown"
