import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_guard as hg  # noqa: E402


class HeavyGuardTests(unittest.TestCase):
    def setUp(self):
        # Never contend with real jobs holding the machine-wide lock.
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(hg, 'LOCK_PATH', Path(self._tmp.name) / 'heavy-jobs.lock')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self._held = os.environ.pop(hg.HELD_ENV, None)
        self.addCleanup(lambda: self._held and os.environ.__setitem__(hg.HELD_ENV, self._held))

    def test_footprint_counts_a_child_allocation(self):
        child = subprocess.Popen([sys.executable, '-c', 'b = bytearray(300 * 1024 * 1024); b[::4096] = b"x" * len(b[::4096]); import time; time.sleep(5)'])
        try:
            for _ in range(50):
                if hg.tree_footprint(os.getpid()) - hg.footprint(os.getpid()) > 250 * 1024 * 1024:
                    break
                subprocess.run(['sleep', '0.1'])
            self.assertGreater(hg.tree_footprint(os.getpid()) - hg.footprint(os.getpid()), 250 * 1024 * 1024)
        finally:
            child.kill()
            child.wait()
        self.assertEqual(hg.footprint(child.pid), 0)

    def test_system_readings_are_sane(self):
        self.assertIn(hg.memory_pressure(), (hg.NORMAL, hg.WARNING, hg.CRITICAL))
        self.assertTrue(0 <= hg.memory_level() <= 100)
        self.assertGreaterEqual(hg.swap_used(), 0)

    def test_system_watch_reports_swap_growth_and_low_memory(self):
        with mock.patch.object(hg, 'swap_used', return_value=0):
            watch = hg.SystemWatch()
        with mock.patch.object(hg, 'memory_pressure', return_value=hg.NORMAL), \
                mock.patch.object(hg, 'memory_level', return_value=50), \
                mock.patch.object(hg, 'swap_used', return_value=5 * hg.GIB):
            self.assertIn('swap grew', watch.problem())
        with mock.patch.object(hg, 'memory_pressure', return_value=hg.NORMAL), \
                mock.patch.object(hg, 'memory_level', return_value=5), \
                mock.patch.object(hg, 'swap_used', return_value=0):
            self.assertIn('available fell', watch.problem())
        with mock.patch.object(hg, 'memory_pressure', return_value=hg.CRITICAL), \
                mock.patch.object(hg, 'memory_level', return_value=50), \
                mock.patch.object(hg, 'swap_used', return_value=0):
            self.assertIn('critical', watch.problem())
        with mock.patch.object(hg, 'memory_pressure', return_value=hg.NORMAL), \
                mock.patch.object(hg, 'memory_level', return_value=50), \
                mock.patch.object(hg, 'swap_used', return_value=0):
            self.assertIsNone(watch.problem())

    def test_lock_is_exclusive_across_processes_and_reentrant_in_children(self):
        probe = ('import fcntl, sys; f = open(sys.argv[1], "a+")\n'
                 'try:\n    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); print("free")\n'
                 'except BlockingIOError:\n    print("held")')
        with mock.patch.object(hg, 'memory_level', return_value=100), \
                mock.patch.object(hg, 'memory_pressure', return_value=hg.NORMAL):
            with hg.HeavyLock(log=lambda m: None):
                out = subprocess.run([sys.executable, '-c', probe, str(hg.LOCK_PATH)],
                                     capture_output=True, text=True).stdout.strip()
                self.assertEqual(out, 'held')
                with hg.HeavyLock(log=lambda m: None):   # nested use in the holder must not deadlock
                    pass
        out = subprocess.run([sys.executable, '-c', probe, str(hg.LOCK_PATH)], capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, 'free')


    def test_start_allows_warning_pressure_but_not_critical_or_low_memory(self):
        with mock.patch.object(hg, 'memory_pressure', return_value=hg.WARNING), \
                mock.patch.object(hg, 'memory_level', return_value=40):
            self.assertIsNone(hg.start_problem())
        with mock.patch.object(hg, 'memory_pressure', return_value=hg.CRITICAL), \
                mock.patch.object(hg, 'memory_level', return_value=40):
            self.assertIn('critical', hg.start_problem())
        with mock.patch.object(hg, 'memory_pressure', return_value=hg.WARNING), \
                mock.patch.object(hg, 'memory_level', return_value=10):
            self.assertIn('available', hg.start_problem())

    def test_wait_out_pauses_until_memory_recovers_and_resets_swap_baseline(self):
        calls = []
        with mock.patch.object(hg, 'swap_used', return_value=0):
            watch = hg.SystemWatch()
        pressure = iter([hg.CRITICAL, hg.CRITICAL, hg.WARNING])
        with mock.patch.object(hg, 'memory_pressure', side_effect=lambda: next(pressure)), \
                mock.patch.object(hg, 'memory_level', return_value=40), \
                mock.patch.object(hg, 'swap_used', return_value=hg.GIB), \
                mock.patch.object(hg.time, 'sleep'):
            hg.SystemWatch.wait_out(watch, log=lambda m: None,
                                    pause=lambda: calls.append('pause'), resume=lambda: calls.append('resume'))
        self.assertEqual(calls, ['pause', 'resume'])
        self.assertEqual(watch.swap_start, hg.GIB)
        with mock.patch.object(hg, 'memory_pressure', return_value=hg.NORMAL), \
                mock.patch.object(hg, 'memory_level', return_value=40), \
                mock.patch.object(hg, 'swap_used', return_value=hg.GIB):
            self.assertEqual(watch.wait_out(log=lambda m: None, pause=lambda: calls.append('x')), 0.0)
        self.assertEqual(calls, ['pause', 'resume'])

    def test_cli_runs_a_command_and_stops_one_that_exceeds_its_limit(self):
        with mock.patch.object(hg, 'memory_level', return_value=100), \
                mock.patch.object(hg, 'memory_pressure', return_value=hg.NORMAL), \
                mock.patch.object(hg, 'swap_used', return_value=0):
            self.assertEqual(hg.run([sys.executable, '-c', 'pass'], log=lambda m: None), 0)
            hog = 'b = bytearray(400 * 1024 * 1024); b[::4096] = b"x" * len(b[::4096]); import time; time.sleep(30)'
            started = __import__('time').monotonic()
            self.assertEqual(hg.run([sys.executable, '-c', hog], limit=200 * 1024 * 1024, log=lambda m: None), 3)
            self.assertLess(__import__('time').monotonic() - started, 20)


if __name__ == '__main__':
    unittest.main()
