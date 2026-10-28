"""Feasibility check: staggered pressure-velocity FDTD on Apple MPS.

1. Rigid shoebox: compare spectral peaks with analytic room modes.
2. Throughput (cell-updates/s) at several grid sizes.
Not a validated reference; it sizes a future wave teacher.
"""
import sys
import time

import numpy as np
import torch

dev = torch.device('mps')
c = 343.0


def run(shape, dx, steps, solid=None, src=None, rec=None):
    nx, ny, nz = shape
    dt = 0.5 * dx / (c * np.sqrt(3))  # Courant 0.5/sqrt(3)
    k = c * dt / dx
    p = torch.zeros(shape, device=dev)
    ux = torch.zeros((nx - 1, ny, nz), device=dev)
    uy = torch.zeros((nx, ny - 1, nz), device=dev)
    uz = torch.zeros((nx, ny, nz - 1), device=dev)
    # Rigid domain walls: velocity arrays exclude the outer faces (zero normal velocity).
    if solid is not None:
        air = (~solid).float()
        mx = air[1:] * air[:-1]
        my = air[:, 1:] * air[:, :-1]
        mz = air[:, :, 1:] * air[:, :, :-1]
    trace = []
    t0 = time.perf_counter()
    for n in range(steps):
        ux -= k * (p[1:] - p[:-1])
        uy -= k * (p[:, 1:] - p[:, :-1])
        uz -= k * (p[:, :, 1:] - p[:, :, :-1])
        if solid is not None:
            ux *= mx; uy *= my; uz *= mz
        div = torch.zeros_like(p)
        div[:-1] += ux; div[1:] -= ux
        div[:, :-1] += uy; div[:, 1:] -= uy
        div[:, :, :-1] += uz; div[:, :, 1:] -= uz
        p -= k * div
        if src is not None and n < 40:
            # Short Gaussian pulse, soft source.
            p[src] += float(-(n - 20) / 6.0 * np.exp(-((n - 20) / 6.0) ** 2))
        if rec is not None:
            trace.append(p[rec].clone())
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    out = torch.stack(trace).cpu().numpy() if trace else None
    return elapsed, dt, out


def modes_check():
    L = np.array([3.0, 2.2, 1.7])
    dx = 0.05
    shape = tuple(int(round(v / dx)) for v in L)
    steps = 40000
    el, dt, tr = run(shape, dx, steps, src=(7, 9, 5), rec=(51, 33, 27))
    np.savez('modes.npz', trace=tr, dt=dt, shape=np.array(shape), dx=dx)
    spec = np.abs(np.fft.rfft(tr * np.hanning(len(tr))))
    f = np.fft.rfftfreq(len(tr), dt)
    sel = (f > 40) & (f < 260)
    fs, ss = f[sel], spec[sel]
    peaks = [i for i in range(1, len(ss) - 1) if ss[i] > ss[i - 1] and ss[i] > ss[i + 1] and ss[i] > 0.02 * ss.max()]
    measured = fs[peaks]
    Lg = np.array(shape) * dx
    analytic = sorted({c / 2 * np.sqrt((a / Lg[0]) ** 2 + (b / Lg[1]) ** 2 + (d / Lg[2]) ** 2)
                       for a in range(6) for b in range(6) for d in range(6) if a + b + d})
    analytic = [a for a in analytic if 40 < a < 260]
    errs = [min(abs(m - a) for a in analytic) / m for m in measured]
    print(f'modes: grid {shape}, {steps} steps in {el:.1f}s; {len(measured)} peaks 40-260 Hz; '
          f'max rel. error to nearest analytic mode {max(errs)*100:.2f}% (freq resolution {f[1]:.2f} Hz)')
    for m in measured[:8]:
        a = min(analytic, key=lambda a: abs(a - m))
        print(f'  measured {m:7.2f} Hz  analytic {a:7.2f} Hz')


def throughput():
    for n in (128, 192, 256, 320):
        steps = 60
        run((n, n, n), 0.05, 5)  # warmup
        el, _, _ = run((n, n, n), 0.05, steps)
        print(f'throughput {n}^3: {n**3*steps/el/1e9:.2f} Gcell/s ({el/steps*1e3:.1f} ms/step)')
        sys.stdout.flush()
        time.sleep(1)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    modes_check()
    throughput()
