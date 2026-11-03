"""Error detection — MCE (Machine Check Exceptions), kernel crash lines and
kernel stall warnings (RCU stalls, soft lockups).

The kernel log is the single source of truth. During a test it is read from
/dev/kmsg (non-blocking, positioned at the end of the buffer when the test
starts), falling back to polling ``dmesg`` when /dev/kmsg cannot be opened. The legacy sysfs
/sys/devices/system/machinecheck/*/bank* files are MCA control registers
(constant enable masks), not error counters — they never change when an
error is logged, so counting them cannot detect anything.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from corecycler.config import tools

log = logging.getLogger(__name__)

# Kernel log device. Module-level so tests can point it elsewhere.
KMSG_PATH = "/dev/kmsg"
# Upper bound on records drained per check, so a log flood cannot pin a poll.
_KMSG_MAX_RECORDS = 10000


@dataclass(slots=True)
class MCEEvent:
    timestamp: float
    cpu: int
    bank: int
    message: str
    corrected: bool
    raw_ts: float = 0.0  # kernel/journal timestamp of the source line
    # "mce" (machine check), "kernel_error" (oops/BUG/stack dump line) or
    # "kernel_stall" (RCU stall, soft lockup). For a stall, cpu is the stalled
    # logical CPU, or -1 when none was named.
    kind: str = "mce"


@dataclass(slots=True)
class ErrorState:
    mce_events: list[MCEEvent] = field(default_factory=list)
    computation_errors: list[str] = field(default_factory=list)
    last_check_time: float = 0.0

    @property
    def has_errors(self) -> bool:
        return bool(self.mce_events or self.computation_errors)


# AMD Zen decoded MCA block: exactly one line carries CPU, bank and severity
# flags together — "[Hardware Error]: CPU:9 (1a:44:0) MC0_STATUS[Over|CE|...]".
# Only that line produces an event; the block's header ("Machine check events
# logged") and detail lines (Corrected error/Error Addr/IPID/Syndrome/unit)
# describe the same event and must not double-count it.
_AMD_STATUS_RE = re.compile(r"cpu[:\s]+(\d+).*?\bmc(\d+)_status\[([^\]]*)\]", re.IGNORECASE)
_CPU_RE = re.compile(r"\bcpu[:\s]+(\d+)", re.IGNORECASE)
_BANK_RE = re.compile(r"\bbank[:\s]+(\d+)", re.IGNORECASE)


# "watchdog: BUG: soft lockup - CPU#3 stuck for 22s! [mprime:1234]"
_SOFT_LOCKUP_RE = re.compile(r"\bsoft lockup\b.*?\bcpu#(\d+)", re.IGNORECASE)
# "rcu: INFO: rcu_preempt detected stalls on CPUs/tasks:" and
# "rcu: INFO: rcu_sched self-detected stall on CPU" (rcu_sched on non-preempt kernels)
_RCU_STALL_RE = re.compile(r"\brcu_\w+ (?:self-)?detected (?:expedited )?stalls?\b", re.IGNORECASE)
# "watchdog: CPU2: Watchdog detected hard LOCKUP on cpu 2". The named CPU is
# often a victim (it spun on a lock the stuck CPU holds), so it is never
# attributed: cpu=-1.
_HARD_LOCKUP_RE = re.compile(r"\bwatchdog detected hard lockup\b", re.IGNORECASE)
# Per-CPU line that follows an RCU stall header: "rcu: \t3-....: (20999 ticks this GP) ..."
# The "(detected by N, ...)" line names the CPU that NOTICED the stall and never matches.
_RCU_STALL_CPU_RE = re.compile(r"^(?:rcu:)?\s+(\d+)-\S*:")
# Older kernels list the CPUs inline: "... detected stalls on CPUs/tasks: { 3-...! 5-... }"
_RCU_INLINE_CPUS_RE = re.compile(r"\{([^}]*)\}")


def _stall_event(body: str) -> MCEEvent | None:
    """A kernel stall warning, or None. RCU headers carry no CPU (cpu=-1);
    the caller attaches the CPUs from the per-CPU lines that follow."""
    m = _SOFT_LOCKUP_RE.search(body)
    if m:
        cpu = int(m.group(1))
    elif _RCU_STALL_RE.search(body) or _HARD_LOCKUP_RE.search(body):
        cpu = -1
    else:
        return None
    return MCEEvent(
        timestamp=time.time(),
        cpu=cpu,
        bank=-1,
        message=body.strip(),
        corrected=False,
        kind="kernel_stall",
    )


def events_from_lines(lines: list[tuple[float, str]]) -> list[MCEEvent]:
    """Classify (raw_ts, body) kernel lines in order into events.

    An RCU stall header is followed by one line per stalled CPU; each named
    CPU becomes its own event. A header with no CPU lines in this batch stays
    a single cpu=-1 event.
    """
    events: list[MCEEvent] = []
    i = 0
    while i < len(lines):
        raw_ts, body = lines[i]
        i += 1
        event = classify_mce_line(body)
        if event is None:
            continue
        event.raw_ts = raw_ts
        if event.kind == "kernel_stall" and event.cpu == -1 and _RCU_STALL_RE.search(body):
            inline = _RCU_INLINE_CPUS_RE.search(body)
            cpus = [int(n) for n in re.findall(r"(\d+)-", inline.group(1))] if inline else []
            while i < len(lines):
                m = _RCU_STALL_CPU_RE.match(lines[i][1])
                if not m:
                    break
                cpus.append(int(m.group(1)))
                i += 1
            for cpu in dict.fromkeys(cpus):
                events.append(
                    MCEEvent(
                        timestamp=event.timestamp,
                        cpu=cpu,
                        bank=-1,
                        message=event.message,
                        corrected=False,
                        raw_ts=raw_ts,
                        kind="kernel_stall",
                    )
                )
            if cpus:
                continue
        events.append(event)
    return events


def classify_mce_line(body: str) -> MCEEvent | None:
    """Classify one kernel message body (no timestamp prefix) into an event.

    Returns None for non-error lines, informational MCE lines, and the
    continuation lines of an AMD decoded block.
    """
    # Stalls first: "watchdog: BUG: soft lockup" would otherwise match the
    # generic "bug:" kernel-error rule and lose its CPU.
    stall = _stall_event(body)
    if stall is not None:
        return stall

    lower = body.lower()

    m = _AMD_STATUS_RE.search(lower)
    if m:
        flags = {t.strip() for t in m.group(3).split("|")}
        corrected = "ce" in flags and "ue" not in flags
        return MCEEvent(
            timestamp=time.time(),
            cpu=int(m.group(1)),
            bank=int(m.group(2)),
            message=body.strip(),
            corrected=corrected,
        )

    if _is_mce_error_line(lower):
        cpu_m = _CPU_RE.search(lower)
        bank_m = _BANK_RE.search(lower)
        corrected = bool(re.search(r"\bcorrected\b", lower)) and "uncorrect" not in lower
        return MCEEvent(
            timestamp=time.time(),
            cpu=int(cpu_m.group(1)) if cpu_m else -1,
            bank=int(bank_m.group(1)) if bank_m else -1,
            message=body.strip(),
            corrected=corrected,
        )

    if _is_kernel_error_line(lower):
        return MCEEvent(
            timestamp=time.time(),
            cpu=-1,
            bank=-1,
            message=body.strip(),
            corrected=False,
            kind="kernel_error",
        )

    return None


class ErrorDetector:
    """Watches the kernel log for hardware errors during stress tests.

    ``check_mce()`` returns every NEW event since ``reset()`` exactly once
    (consume semantics) so callers can both fail the current test on the
    tested core's events and record other cores' events as evidence without
    re-processing duplicates on each poll.
    """

    # Minimum interval between dmesg subprocess calls (seconds). The
    # /dev/kmsg path has no rate limit: a read is a cheap non-blocking syscall.
    DMESG_MIN_INTERVAL: float = 5.0
    # A stall report is followed by the stalled CPU's stack dump ("RIP: ...",
    # "Call Trace:"), which matches the kernel-error rule. Kernel-error lines
    # logged within this many seconds (kernel timestamps) after a stall belong
    # to that report and are dropped, even when they arrive in a later poll.
    STALL_TRACE_WINDOW: float = 10.0

    def __init__(self) -> None:
        self._dmesg_baseline_ts: float = 0.0  # raw monotonic timestamp
        self._last_dmesg_time: float = 0.0
        self._seen: set[tuple[float, int, int]] = set()
        self._kmsg_fd: int | None = None
        self._last_stall_ts: float | None = None

    def reset(self) -> None:
        """Start a new observation window — only lines newer than now count."""
        self.close()
        self._seen.clear()
        self._last_stall_ts = None
        self._last_dmesg_time = 0.0
        self._kmsg_fd = _open_kmsg_at_end(KMSG_PATH)
        if self._kmsg_fd is None:
            self._dmesg_baseline_ts = _get_dmesg_raw_timestamp()

    def close(self) -> None:
        """Release the /dev/kmsg descriptor (reset() reopens it)."""
        if self._kmsg_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._kmsg_fd)
            self._kmsg_fd = None

    def __del__(self) -> None:
        self.close()

    def check_mce(self) -> list[MCEEvent]:
        """Return new MCE/kernel-error/kernel-stall events since reset(), each
        exactly once.

        From /dev/kmsg: every record logged since reset(), on every call.
        From the dmesg fallback: rate-limited — within DMESG_MIN_INTERVAL of
        the previous subprocess call it returns [] instead of re-running
        dmesg; the event is delivered on the first poll after the interval.
        """
        events = self._accept(events_from_lines(self._read_lines()))
        if self._kmsg_fd is None and events:
            log.debug(
                "dmesg poll: %d new event(s)%s",
                len(events),
                "".join(f" [cpu={e.cpu} bank={e.bank} ce={e.corrected}]" for e in events),
            )
        return events

    def _read_lines(self) -> list[tuple[float, str]]:
        """New (raw_ts, body) kernel lines since the last call."""
        if self._kmsg_fd is not None:
            return _drain_kmsg(self._kmsg_fd)

        now = time.monotonic()
        if now - self._last_dmesg_time < self.DMESG_MIN_INTERVAL:
            return []
        self._last_dmesg_time = now

        # No baseline means old and new lines are indistinguishable — treating
        # boot-time history as fresh errors would fail every first test.
        if self._dmesg_baseline_ts <= 0:
            return []

        lines: list[tuple[float, str]] = []
        try:
            # No --level filter: AMD decoded corrected-error lines are logged
            # below err/warn and a level filter silently hides them; the line
            # classifier is the filter.
            result = subprocess.run(
                [tools.command_name("dmesg"), "--time-format=raw"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []
            for line in result.stdout.splitlines():
                ts_match = re.match(r"\s*\[?\s*([\d.]+)\]?\s?", line)
                if not ts_match:
                    continue
                try:
                    raw_ts = float(ts_match.group(1))
                except ValueError:
                    continue
                if raw_ts <= self._dmesg_baseline_ts:
                    continue
                lines.append((raw_ts, line[ts_match.end() :]))
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, PermissionError) as exc:
            log.debug("dmesg check failed: %s", exc)
        return lines

    def _accept(self, events: list[MCEEvent]) -> list[MCEEvent]:
        """Drop events already returned and the stack-dump lines of a stall report."""
        fresh: list[MCEEvent] = []
        for event in events:
            key = (event.raw_ts, event.cpu, event.bank)
            if key in self._seen:
                continue
            self._seen.add(key)
            if event.kind == "kernel_stall":
                self._last_stall_ts = max(self._last_stall_ts or event.raw_ts, event.raw_ts)
            elif (
                event.kind == "kernel_error"
                and self._last_stall_ts is not None
                and 0 <= event.raw_ts - self._last_stall_ts <= self.STALL_TRACE_WINDOW
            ):
                log.debug("dropped stall stack-dump line: %s", event.message)
                continue
            fresh.append(event)
        return fresh


def _open_kmsg_at_end(path: str) -> int | None:
    """Open the kernel log device positioned AFTER every record already in
    the buffer, so only messages logged from now on are read."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as exc:
        log.debug("%s unavailable (%s) — falling back to dmesg polling", path, exc)
        return None
    try:
        os.lseek(fd, 0, os.SEEK_END)
    except OSError as exc:
        log.debug("cannot seek %s (%s) — falling back to dmesg polling", path, exc)
        os.close(fd)
        return None
    return fd


_KMSG_ESCAPE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")


def parse_kmsg_record(record: bytes) -> tuple[float, str] | None:
    """Parse one /dev/kmsg record "prio,seq,ts_usec,flags;text\n KEY=VAL..."
    into (seconds since boot, text). Continuation lines (device metadata)
    are dropped; the kernel's \\xNN escapes (a tab arrives as \\x09) are undone."""
    text = record.decode("utf-8", errors="replace")
    header, sep, rest = text.partition(";")
    if not sep:
        return None
    fields = header.split(",")
    if len(fields) < 3:
        return None
    try:
        ts = int(fields[2]) / 1_000_000
    except ValueError:
        return None
    message = rest.split("\n", 1)[0]
    return ts, _KMSG_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), message)


def _drain_kmsg(fd: int) -> list[tuple[float, str]]:
    """Read every record available on a non-blocking /dev/kmsg descriptor."""
    lines: list[tuple[float, str]] = []
    for _ in range(_KMSG_MAX_RECORDS):
        try:
            record = os.read(fd, 8192)
        except BlockingIOError:
            break  # drained
        except BrokenPipeError:
            continue  # records were overwritten before we read them; resume with the next
        except OSError as exc:
            log.debug("/dev/kmsg read failed: %s", exc)
            break
        if not record:
            break
        parsed = parse_kmsg_record(record)
        if parsed is not None:
            lines.append(parsed)
    return lines


def harvest_kernel_mce(since_utc_iso: str, timeout: float = 15.0) -> tuple[list[MCEEvent], bool]:
    """Read MCE/kernel-error events from the systemd journal since a UTC ISO
    timestamp — across reboots, so it covers the boot(s) a hard crash killed.

    Returns (events, harvest_ok). harvest_ok False means the journal could
    not be read at all (journalctl missing, unreadable, bad timestamp) — the
    caller must treat the crash as unattributed, never as a clean bill.
    """
    since = _iso_to_journal_since(since_utc_iso)
    if since is None:
        return [], False
    try:
        # _TRANSPORT=kernel, NOT -k: -k implies --boot (current boot only),
        # which silently hides the crashed boot's MCE lines — the entire
        # point of this harvest.
        result = subprocess.run(
            [
                tools.command_name("journalctl"),
                "-q",
                "--no-pager",
                "-o",
                "short-unix",
                "--since",
                since,
                "_TRANSPORT=kernel",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, PermissionError) as exc:
        log.warning("kernel-journal harvest failed: %s", exc)
        return [], False
    if result.returncode != 0:
        log.warning(
            "journalctl exited %d during harvest: %s",
            result.returncode,
            result.stderr.strip()[:200],
        )
        return [], False

    events: list[MCEEvent] = []
    seen: set[tuple[float, int, int]] = set()
    for line in result.stdout.splitlines():
        # short-unix format: "1626382113.123456 host kernel: <message>"
        m = re.match(r"\s*([\d.]+)\s+\S+\s+kernel:\s?(.*)$", line)
        if not m:
            continue
        event = classify_mce_line(m.group(2))
        if event is None:
            continue
        if event.kind == "kernel_stall":
            # Post-crash forensics keep their old behaviour: a soft lockup is
            # an unattributed kernel error (it matched "bug:" before stall
            # detection existed); an RCU stall line was never an event.
            if not _SOFT_LOCKUP_RE.search(event.message):
                continue
            event.kind = "mce"
            event.cpu = -1
        try:
            event.raw_ts = float(m.group(1))
        except ValueError:
            event.raw_ts = 0.0
        key = (event.raw_ts, event.cpu, event.bank)
        if key in seen:
            continue
        seen.add(key)
        events.append(event)
    return events, True


def last_boot_ended_cleanly(timeout: float = 15.0) -> bool:
    """True when the previous boot ended in an orderly shutdown.

    Journald writes a final "Journal stopped" record on every clean shutdown;
    a freeze or hard reset leaves the boot's journal without one. Fail closed:
    any read problem returns False — an unproven shutdown is treated as dirty.
    """
    try:
        result = subprocess.run(
            [
                tools.command_name("journalctl"),
                "-q",
                "--no-pager",
                "-b",
                "-1",
                "-n",
                "25",
                "-o",
                "cat",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, PermissionError) as exc:
        log.warning("clean-shutdown probe failed: %s", exc)
        return False
    if result.returncode != 0:
        return False
    return "Journal stopped" in result.stdout or "Shutting down." in result.stdout


def _iso_to_journal_since(iso_ts: str) -> str | None:
    """Convert an ISO-8601 timestamp to systemd's --since form, in UTC."""
    try:
        dt = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _is_mce_error_line(line_lower: str) -> bool:
    """Return True if a dmesg line is an actual MCE error, not informational.

    Excludes boot/info messages like:
    - "mce: CPU supports N MCE banks"
    - "Machine check events logged" (the AMD block header — the event itself
      is counted from the MCx_STATUS line of the same block)
    - "mce_cpu_quirks"
    - "mce: [Hardware Error]:" informational lines about MCE configuration
    """
    # Must mention MCE or machine check at all
    if "mce" not in line_lower and "machine check" not in line_lower:
        return False

    # Exclude known informational patterns
    info_patterns = [
        "cpu supports",
        "mce banks",
        "events logged",
        "mce_cpu_quirks",
        "mce: using",
        "mce: cmci",
        "mce_intel",
        "check polling timer",
    ]
    for pat in info_patterns:
        if pat in line_lower:
            return False

    # Real MCE error indicators
    error_indicators = [
        "hardware error",
        "bank ",  # "Bank N:" with actual bank data
        "machine check exception",
        "fatal machine check",
        "uncorrected",
        "corrected error",
        "mca:",
        "status:",
        "tsc:",
        "addr:",
        "misc:",
        "severity:",
    ]
    return any(ind in line_lower for ind in error_indicators)


def _is_kernel_error_line(line_lower: str) -> bool:
    """Return True if a dmesg line indicates a kernel crash, oops, or BUG.

    CO undervolting can cause kernel-level faults that manifest as oops or
    BUG traps rather than MCE events — especially under heavy instruction
    pressure with AVX/SSE workloads.
    """
    indicators = [
        "kernel panic",
        "oops:",
        "bug:",
        "bug at ",
        "general protection fault",
        "invalid opcode",
        "rip:",
        "call trace:",
        "kernel bug at",
    ]
    return any(ind in line_lower for ind in indicators)


def _get_dmesg_raw_timestamp() -> float:
    """Get the latest dmesg raw monotonic timestamp for baseline filtering."""
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError, OSError, PermissionError):
        result = subprocess.run(
            [tools.command_name("dmesg"), "--time-format=raw"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.strip().splitlines()
        if lines:
            ts_str = lines[-1].split()[0] if lines[-1] else ""
            try:
                return float(ts_str.strip("[]"))
            except ValueError:
                return 0.0
    return 0.0
