"""Machine-wide guard for heavy Unity/GPU jobs on a shared machine: the 16 GB Mac, or the gaming PC
through WSL, where the jobs are Windows processes (Unity.exe, the players) watched through one
persistent PowerShell session (see "Windows, from WSL" below).

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
WSL = 'microsoft' in os.uname().release.lower()
# Shared by every project on the machine; on the PC, /tmp/unity.lock is the lab's Unity slot.
LOCK_PATH = Path('/tmp/unity.lock') if WSL else Path.home() / '.cache' / 'heavy-jobs.lock'
HELD_ENV = 'HEAVY_JOB_LOCK_HELD'          # set for children of a lock holder
# Percent of memory available: kern.memorystatus_level on the Mac; on the PC the commit headroom over its 32 GB of
# physical memory (see memory_level), where 12% is about 3.8 GB, 1.7 times the largest job's footprint (2.24 GiB, the showcase build), and 6% about 2 GB.
MIN_START_LEVEL = 12 if WSL else 25
MIN_RUN_LEVEL = 6 if WSL else 12
NORMAL, WARNING, CRITICAL = 1, 2, 4      # kern.memorystatus_vm_pressure_level
MAX_SWAP_GROWTH = 4 * GIB
WORK_FRACTION = .5                       # default pacing: work half the time, idle half

_libproc = None if WSL else ctypes.CDLL('/usr/lib/libproc.dylib')


class _Usage(ctypes.Structure):
    # Prefix of rusage_info_v0 from <sys/resource.h>; flavor 0 fills exactly this.
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins', 'wired_size',
        'resident_size', 'phys_footprint', 'proc_start_abstime', 'proc_exit_abstime')]


def _proc_stat(pid):
    """Fields of /proc/<pid>/stat after the command name (Linux), or None if the process is gone."""
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None


def footprint(pid):
    """Physical footprint in bytes (on the Mac including GPU memory; in WSL the resident set); 0 if the process is gone."""
    if WSL:
        stat = _proc_stat(pid)
        return int(stat[21]) * os.sysconf('SC_PAGE_SIZE') if stat else 0
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


def start_time(pid):
    """Kernel start time of `pid`, or None if it is gone (guards against PID reuse)."""
    if WSL:
        stat = _proc_stat(pid)
        return int(stat[19]) if stat else None
    usage = _Usage()
    return usage.proc_start_abstime if _libproc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0 else None


# ---------------------------------------------------------------- Windows, from WSL
# On the PC the jobs are Windows processes: WSL sees only the interop proxy, so a job's tree is found by a marker
# in its command line (its -logFile, unique per run), its footprint is the Windows commit (PrivatePageCount), and
# pausing is NtSuspendProcess. One persistent PowerShell answers every query (~0.25 s each; a fresh one takes 4 s).

WINDOWS_WORK = Path('/mnt/c/Users/clark/aw/audio-raytracing')   # the Windows drive: Unity's runner, builds
# Each worktree gets its own runner and builds there, so parallel worktrees never re-import each other's code; the
# main worktree (audio-raytracing) keeps the top level, whose runner seeds a new one's Library (dlss_jobs.sync_runner).
_WORKTREE = Path(__file__).resolve().parents[1].name
WORKTREE_WORK = WINDOWS_WORK if _WORKTREE == 'audio-raytracing' else WINDOWS_WORK / 'worktrees' / _WORKTREE
_POWERSHELL = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
_SERVER = r"""
Add-Type -Namespace Nt -Name P -MemberDefinition '[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h); [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
while ($true) {
  $line = [Console]::In.ReadLine(); if ($line -eq $null) { break }
  $verb, $arg = $line -split ' ', 2
  try {
    if ($verb -eq 'snap') {
      $os = Get-CimInstance Win32_OperatingSystem
      $m = $os.FreeVirtualMemory; $t = $os.TotalVisibleMemorySize
      $b = @(Get-CimInstance Win32_Battery)[0]
      $all = Get-CimInstance Win32_Process -Property ProcessId,ParentProcessId,Name,PrivatePageCount,CommandLine,CreationDate
      $p = @($all | % { ,@($_.ProcessId, $_.ParentProcessId, [int64]$_.PrivatePageCount, $_.CreationDate.Ticks, [int]($arg -and $_.CommandLine -and $_.CommandLine.Contains($arg))) })
      $u = @($all | ? { $_.Name -eq 'Unity.exe' } | % { $_.ProcessId })
      $r = @{commit=$m; total=$t; ac=$(if ($b) { [int]($b.BatteryStatus -ne 1) } else { 1 }); charge=$(if ($b) { $b.EstimatedChargeRemaining } else { 100 }); p=$p; unity=$u}
    } else {
      foreach ($id in ($arg -split ',')) { try {   # a process may exit between the lookup and the action
        $h = Get-Process -Id ([int]$id) -ErrorAction Stop
        if ($verb -eq 'kill') { Stop-Process -Id $h.Id -Force -ErrorAction SilentlyContinue }
        elseif ($verb -eq 'lower') { $h.PriorityClass = 'BelowNormal' }
        elseif ($verb -eq 'suspend') { [void][Nt.P]::NtSuspendProcess($h.Handle) }
        elseif ($verb -eq 'resume') { [void][Nt.P]::NtResumeProcess($h.Handle) }
      } catch {} }
      $r = @{ok=1}
    }
  } catch { $r = @{error="$_"} }
  [Console]::Out.WriteLine(($r | ConvertTo-Json -Compress -Depth 4)); [Console]::Out.Flush()
}
"""


class _Windows:
    def __init__(self):
        script = WINDOWS_WORK / 'heavy_guard.ps1'
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(_SERVER)
        windows_script = subprocess.run(['wslpath', '-w', str(script)], capture_output=True, text=True, check=True).stdout.strip()
        self.process = subprocess.Popen([_POWERSHELL, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                                         '-File', windows_script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, start_new_session=True)

    def ask(self, verb, argument=''):
        import json
        self.process.stdin.write(f'{verb} {argument}\n')
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if 'error' in reply:
            raise RuntimeError(f'PowerShell {verb}: {reply["error"]}')
        return reply

    def snapshot(self, marker=''):
        """Available MB, total bytes, AC power, every process as (pid, ppid, commit bytes, start ticks, marked),
        the PIDs of every Unity.exe."""
        return self.ask('snap', marker)


_windows = None


def windows():
    global _windows
    if _windows is None or _windows.process.poll() is not None:
        _windows = _Windows()
    return _windows


class WindowsTree:
    """The Windows processes of one job: those whose command line contains `marker`, and their descendants."""
    POLL_SECONDS = 2.0

    def __init__(self, marker):
        self.marker, self.seen, self.last = marker, {}, {}

    def footprint(self):
        rows = {pid: (parent, used, started, marked) for pid, parent, used, started, marked in windows().snapshot(self.marker)['p']}
        members = {pid for pid, row in rows.items() if row[3]}
        frontier = set(members)
        while frontier:
            frontier = {pid for pid, row in rows.items() if row[0] in frontier and pid not in members}
            members |= frontier
        new = [pid for pid in members if pid not in self.seen]
        for pid in new:
            self.seen[pid] = rows[pid][2]
        if new:
            windows().ask('lower', ','.join(map(str, new)))   # below-normal priority, Windows' nice
        self.last = {pid: row[2] for pid, row in rows.items()}
        return sum(rows[pid][1] for pid in self.alive())

    def alive(self):
        return [pid for pid, started in self.seen.items() if self.last.get(pid) == started]

    def signal(self, number):
        verb = {signal.SIGSTOP: 'suspend', signal.SIGCONT: 'resume'}.get(number, 'kill')
        if self.alive():
            windows().ask(verb, ','.join(map(str, self.alive())))

    def kill(self, grace=10):
        self.footprint()   # pick up helpers started since the last poll
        self.signal(signal.SIGCONT)
        self.signal(signal.SIGKILL)


def job_tree(process, command):
    """The processes of a job started as `process` from `command`: on the PC its Windows processes (matched by the
    command's -logFile), on the Mac its process tree."""
    if WSL and '-logFile' in command:
        return WindowsTree(command[command.index('-logFile') + 1])
    return ProcessTree(process.pid)


def host_command(command, env):
    """`command` and `env` for this machine: on the PC, POSIX paths become Windows paths (wslpath -w) and the
    environment's ACOUSTIC_, HEAVY_ and UNITY_ variables are passed through to Windows (WSLENV). Elsewhere, unchanged."""
    if WSL and command[:2] == ['nice', '-n']:   # Windows' nice is the below-normal priority WindowsTree sets
        command = command[3:]
    if not WSL or not command[0].endswith('.exe'):
        return command, env
    def native(a):   # Unity's options start with '-', so an argument starting with '/' is a path
        if a.startswith('/'):
            return subprocess.run(['wslpath', '-w', a], capture_output=True, text=True, check=True).stdout.strip()
        return a
    passed = [k for k in env if k.startswith(('ACOUSTIC_', 'HEAVY_', 'UNITY_'))]
    return [command[0]] + [native(a) for a in command[1:]], dict(env, WSLENV=':'.join(filter(None, [env.get('WSLENV', '')] + passed)))


class ProcessTree:
    """Every process a job has spawned, including helpers that start their own process
    group or are reparented when their parent dies (Unity's shader compiler does both)."""

    POLL_SECONDS = .5

    def __init__(self, root):
        self.root, self.seen = root, {}

    def alive(self):
        return [p for p, started in self.seen.items() if start_time(p) == started]

    def footprint(self):
        for p in tree(self.root):
            if p not in self.seen and (started := start_time(p)) is not None:
                self.seen[p] = started
        return sum(footprint(p) for p in self.alive())

    def signal(self, number):
        for p in self.alive():
            try:
                os.kill(p, number)
            except (ProcessLookupError, PermissionError):
                pass

    def kill(self, grace=10):
        self.signal(signal.SIGCONT)
        self.signal(signal.SIGTERM)
        deadline = time.monotonic() + grace
        while self.alive() and time.monotonic() < deadline:
            time.sleep(.2)
        self.signal(signal.SIGKILL)


def memory_level():
    if WSL:   # Windows pages rather than fails while commit lasts, and the WSL VM keeps physical memory near full,
        snap = windows().snapshot()   # so the level is commit headroom (KB) over physical memory (KB)
        return min(100, int(100 * snap['commit'] / snap['total']))
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_level'],
                              capture_output=True, text=True).stdout)


def memory_pressure():
    if WSL:   # Windows has no pressure level; almost no commit headroom is critical
        return CRITICAL if memory_level() < 3 else NORMAL
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                              capture_output=True, text=True).stdout)


def swap_used():
    if WSL:   # the page file's growth is not tracked on Windows
        return 0
    text = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    value, unit = re.search(r'used = ([\d.]+)([MG])', text).groups()
    return int(float(value) * (GIB if unit == 'G' else 1024 ** 2))


CRITICAL_REASON = 'Windows commit headroom is critical (under 3%)' if WSL else 'memory pressure is critical'
QUIET_ENV = 'HEAVY_QUIET'   # set for GPU timing measurements: they start only on a quiet GPU
GATE, GATED_ENV = Path.home() / '.agentworld/bin/gpu', 'HEAVY_GPU_GATED'


def gpu_gate():
    """On the PC, rerun this command under the lab's GPU gate (~/.agentworld/bin/gpu): alone for a timing run
    (HEAVY_QUIET), else beside other jobs at a Unity editor's size. Call it first in a tool's main."""
    if not WSL or os.environ.get(GATED_ENV) or os.environ.get('AGENTWORLD_GPU_JOB') or not GATE.exists() or {'-h', '--help'} & set(sys.argv):
        return
    os.environ[GATED_ENV] = '1'
    size = ['--alone'] if os.environ.get(QUIET_ENV) else ['--vram', '2G', '--ram', '4G']
    os.execv(str(GATE), [str(GATE), *size, sys.executable, *sys.argv])


def gpu_busy():
    """Why the PC's GPU is not quiet enough to time on (another Unity running, or the GPU over 10% busy), else None.
    Other agents' Unity jobs do not take the lab's lock, so a measurement waits for them to finish."""
    if others := windows().snapshot()['unity']:
        return f'another Unity is running (PID {", ".join(map(str, others))})'
    smi = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    if smi.isdigit() and int(smi) > 10:
        return f'the GPU is {smi}% busy'
    return None


def start_problem():
    """Reason a heavy job should not start yet, else None. Warning pressure is this
    machine's everyday state, so starting needs non-critical pressure and free memory."""
    if memory_pressure() >= CRITICAL:
        return CRITICAL_REASON
    if WSL and os.environ.get(QUIET_ENV) and (busy := gpu_busy()):
        return busy
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
            return CRITICAL_REASON
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
                self.log(f'holding {LOCK_PATH}: starting')
                break
            self._release()   # memory went away while queued; let others try
        os.environ[HELD_ENV] = '1'
        return self

    def _wait_for_memory(self):
        while reason := start_problem():
            self.log(f'waiting to start: {reason}; close other apps or wait')
            time.sleep(self.poll)

    def _release(self):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()
        self.file = None

    def __exit__(self, *exc):
        if self.file is not None:
            os.environ.pop(HELD_ENV, None)
            self._release()
        return False


def run(command, limit=3 * GIB, log=print):
    """Run `command` at nice 19 under the lock; stop it if its tree or the system runs out of memory."""
    with HeavyLock(log):
        watch = SystemWatch()
        process = subprocess.Popen(['nice', '-n', '19'] + command, start_new_session=True,
                                   env=dict(os.environ, **{HELD_ENV: '1'}))
        jobs = ProcessTree(process.pid)
        peak, reason = 0, None
        try:
            while process.poll() is None:
                used = jobs.footprint()
                peak = max(peak, used)
                if used > limit:   # the job itself is too big: stop it
                    reason = f'footprint {used / GIB:.2f} GiB exceeded {limit / GIB:.1f} GiB'
                    break
                watch.wait_out(log, pause=lambda: jobs.signal(signal.SIGSTOP),
                               resume=lambda: jobs.signal(signal.SIGCONT))
                time.sleep(ProcessTree.POLL_SECONDS)
        finally:
            jobs.kill()   # also reaps helpers left behind by a finished or stopped job
            process.wait()
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
