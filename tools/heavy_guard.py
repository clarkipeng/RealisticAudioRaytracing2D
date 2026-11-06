"""Machine-wide guard for heavy Unity/GPU jobs on a shared 16 GB Mac.

`ps` RSS omits Metal/MPS allocations, so an RSS limit cannot see GPU memory.
`phys_footprint` (Activity Monitor's "Memory" column) includes it. Every heavy
job takes one machine-wide lock, so FDTD and Unity runs never overlap, and stops
when the system itself runs short of memory, not only when its own tree is big.
"""
import ctypes
import fcntl
import os
import re
import subprocess
import time
from pathlib import Path

GIB = 1024 ** 3
LOCK_PATH = Path.home() / '.cache' / 'acoustics-heavy-job.lock'
HELD_ENV = 'ACOUSTICS_HEAVY_LOCK_HELD'   # set for children of a lock holder
MIN_START_LEVEL = 25                     # kern.memorystatus_level: percent of memory available
MIN_RUN_LEVEL = 12
MAX_SWAP_GROWTH = 1 * GIB

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


def swap_used():
    text = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    value, unit = re.search(r'used = ([\d.]+)([MG])', text).groups()
    return int(float(value) * (GIB if unit == 'G' else 1024 ** 2))


class SystemWatch:
    """Reason to stop when the whole machine is short of memory, else None."""

    def __init__(self):
        self.swap_start = swap_used()

    def problem(self):
        level = memory_level()
        if level < MIN_RUN_LEVEL:
            return f'system memory available fell to {level}%'
        grown = swap_used() - self.swap_start
        if grown > MAX_SWAP_GROWTH:
            return f'swap grew {grown / GIB:.1f} GiB during the job'
        return None


class HeavyLock:
    """Hold the machine-wide heavy-job lock, then wait until memory is available."""

    def __init__(self, log=print, poll_seconds=30):
        self.log, self.poll, self.file = log, poll_seconds, None

    def __enter__(self):
        if os.environ.get(HELD_ENV):
            return self
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(LOCK_PATH, 'a+')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.log(f'waiting for another heavy job ({LOCK_PATH})')
            fcntl.flock(self.file, fcntl.LOCK_EX)
        while (level := memory_level()) < MIN_START_LEVEL:
            self.log(f'waiting: system memory available {level}% < {MIN_START_LEVEL}%')
            time.sleep(self.poll)
        os.environ[HELD_ENV] = '1'
        return self

    def __exit__(self, *exc):
        if self.file is not None:
            os.environ.pop(HELD_ENV, None)
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None
        return False
