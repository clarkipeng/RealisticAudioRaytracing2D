import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heavy_guard as hg  # noqa: E402


class HeavyGuardTests(unittest.TestCase):
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
                mock.patch.object(hg, 'swap_used', return_value=2 * hg.GIB):
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
