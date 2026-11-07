"""Machine-wide guard for heavy Unity/GPU jobs on a shared 16 GB Mac.

Any project can wrap a command with it (standard library only):

    python3 heavy_guard.py run --limit-gib 3 -- <command> [args...]

A heavy job starts when memory pressure is not critical and 25% is available. It is paused
(not killed) while pressure is critical, available memory is below 12% or swap has grown
4 GiB, and resumes when memory recovers. It is stopped only if its own footprint passes
its limit.

`ps` RSS omits Metal/MPS allocations, so an RSS limit cannot see GPU memory.
`phys_footprint` (Activity Monitor's "Memory" column) includes it. Every heavy
job takes one machine-wide lock, so FDTD and Unity runs never overlap, and stops
when the system itself runs short of memory, not only when its own tree is big.
"""
import argparse
import ctypes
import fcntl
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024 ** 3
LOCK_PATH = Path.home() / '.cache' / 'heavy-jobs.lock'   # shared by every project on this Mac
HELD_ENV = 'HEAVY_JOB_LOCK_HELD'          # set for children of a lock holder
MIN_START_LEVEL = 25                     # kern.memorystatus_level: percent of memory available
MIN_RUN_LEVEL = 12
NORMAL, WARNING, CRITICAL = 1, 2, 4      # kern.memorystatus_vm_pressure_level
MAX_SWAP_GROWTH = 4 * GIB
WORK_FRACTION = .5                       # default pacing: work half the time, idle half

_libproc = ctypes.CDLL('/usr/lib/libproc.dylib')


class _Usage(ctypes.Structure):
    # Prefix of rusage_info_v0 from <sys/resource.h>; flavor 0 fills exactly this.
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins', 'wired_size',
        'resident_size', 'phys_footprint', 'proc_start_abstime', 'proc_exit_abstime')]


def footprint(pid):
    """Physical footprint in bytes, including GPU memory; 0 if the process is gone."""
    usage = _Usage()
    return usage.phys_footprint if _libproc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0 else 0


def tree(pid):
    out = subprocess.run(['ps', '-A', '-o', 'pid=,ppid='], capture_output=True, text=True).stdout.split()
    pairs = list(zip(map(int, out[0::2]), map(int, out[1::2])))
    members, frontier = {pid}, [pid]
    while frontier:
        frontier = [child for child, parent in pairs if parent in frontier and child not in members]
        members.update(frontier)
    return members


def tree_footprint(pid):
    return sum(footprint(p) for p in tree(pid))


def memory_level():
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_level'],
                              capture_output=True, text=True).stdout)


def memory_pressure():
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                              capture_output=True, text=True).stdout)


def swap_used():
    text = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    value, unit = re.search(r'used = ([\d.]+)([MG])', text).groups()
    return int(float(value) * (GIB if unit == 'G' else 1024 ** 2))


def start_problem():
    """Reason a heavy job should not start yet, else None. Warning pressure is this
    machine's everyday state, so starting needs non-critical pressure and free memory."""
    if memory_pressure() >= CRITICAL:
        return 'macOS memory pressure is critical'
    level = memory_level()
    if level < MIN_START_LEVEL:
        return f'system memory available {level}% < {MIN_START_LEVEL}%'
    return None


class SystemWatch:
    """Reason to stop when the whole machine is short of memory, else None."""

    def __init__(self):
        self.swap_start = swap_used()

    def problem(self):
        if memory_pressure() >= CRITICAL:
            return 'macOS memory pressure is critical'
        level = memory_level()
        if level < MIN_RUN_LEVEL:
            return f'system memory available fell to {level}%'
        grown = swap_used() - self.swap_start
        if grown > MAX_SWAP_GROWTH:
            return f'swap grew {grown / GIB:.1f} GiB during the job'
        return None

    def wait_out(self, log=print, pause=None, resume=None, poll=15):
        """Pause, rather than kill, while the machine is short of memory: other apps usually
        cause it and killing a small job frees nothing. Returns the seconds spent paused."""
        problem = self.problem()
        if not problem:
            return 0.0
        log(f'pausing: {problem}')
        if pause:
            pause()
        started = time.monotonic()
        while start_problem() is not None:
            time.sleep(poll)
        if resume:
            resume()
        self.swap_start = swap_used()
        log(f'resumed after {time.monotonic() - started:.0f} s')
        return time.monotonic() - started


class HeavyLock:
    """Hold the machine-wide heavy-job lock, then wait until memory is available."""

    def __init__(self, log=print, poll_seconds=30):
        self.log, self.poll, self.file = log, poll_seconds, None

    def __enter__(self):
        if os.environ.get(HELD_ENV):
            return self
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        while True:
            self._wait_for_memory()            # never hold the lock while waiting for memory
            self.file = open(LOCK_PATH, 'a+')
            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.log(f'waiting for another heavy job ({LOCK_PATH})')
                fcntl.flock(self.file, fcntl.LOCK_EX)
            if start_problem() is None:
                break
            fcntl.flock(self.file, fcntl.LOCK_UN)   # memory went away while queued; let others try
            self.file.close()
        os.environ[HELD_ENV] = '1'
        return self

    def _wait_for_memory(self):
        while reason := start_problem():
            self.log(f'waiting to start: {reason}; close other apps or wait')
            time.sleep(self.poll)

    def __exit__(self, *exc):
        if self.file is not None:
            os.environ.pop(HELD_ENV, None)
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None
        return False


def run(command, limit=3 * GIB, log=print):
    """Run `command` at nice 19 under the lock; stop it if its tree or the system runs out of memory."""
    with HeavyLock(log):
        watch = SystemWatch()
        process = subprocess.Popen(['nice', '-n', '19'] + command, start_new_session=True,
                                   env=dict(os.environ, **{HELD_ENV: '1'}))
        group = lambda number: lambda: os.killpg(process.pid, number)
        peak, reason = 0, None
        try:
            while process.poll() is None:
                used = tree_footprint(process.pid)
                peak = max(peak, used)
                if used > limit:   # the job itself is too big: stop it
                    reason = f'footprint {used / GIB:.2f} GiB exceeded {limit / GIB:.1f} GiB'
                    break
                watch.wait_out(log, pause=group(signal.SIGSTOP), resume=group(signal.SIGCONT))
                time.sleep(2)
        finally:
            if process.poll() is None:
                for number, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, None)):
                    try:
                        os.killpg(process.pid, number)
                    except (PermissionError, ProcessLookupError):
                        process.send_signal(number)
                    try:
                        process.wait(timeout=wait)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        log(f'heavy_guard: {"stopped: " + reason if reason else f"exit {process.returncode}"}; '
            f'peak footprint {peak / GIB:.2f} GiB')
        return 3 if reason else process.returncode


def main():
    parser = argparse.ArgumentParser(description='Run one heavy job at a time on this Mac.')
    sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run', help='run a command under the shared lock and memory guard')
    r.add_argument('--limit-gib', type=float, default=3.0)
    r.add_argument('command', nargs=argparse.REMAINDER)
    sub.add_parser('status', help='show memory pressure, swap and whether the lock is held')
    args = parser.parse_args()
    if args.action == 'status':
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCK_PATH, 'a+') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = False
            except BlockingIOError:
                held = True
        names = {NORMAL: 'normal', WARNING: 'warning', CRITICAL: 'critical'}
        print(f'pressure {names.get(memory_pressure(), "?")}, {memory_level()}% available, '
              f'swap {swap_used() / GIB:.1f} GiB, lock {"held" if held else "free"} ({LOCK_PATH})')
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('give a command after --')
    return run(command, int(args.limit_gib * GIB))


if __name__ == '__main__':
    sys.exit(main())
