#!/usr/bin/env python3
"""Create the original dry, uncalibrated ground-impact kernel used by the demo.

This is a short noise transient, not a recording of rain or a material/modal model.
Emission scheduling, spatial placement and propagation belong to the renderer.
"""
import math
from pathlib import Path
import random
import struct
import wave


def main():
    rate = 44100
    randomizer = random.Random(173)
    samples, low = [], 0.0
    for i in range(round(rate * .024)):
        t = i / rate
        noise = randomizer.uniform(-1, 1)
        low += .25 * (noise - low)
        envelope = (1 - math.exp(-t / .00025)) * math.exp(-t / .0035)
        envelope *= .5 + .5 * math.cos(math.pi * i / (round(rate * .024) - 1))
        samples.append((noise - .65 * low) * envelope)
    # Zero integral and endpoints without a boundary discontinuity.
    window = [math.sin(math.pi * i / (len(samples) - 1)) ** 2 for i in range(len(samples))]
    dc = sum(samples) / sum(window)
    samples = [x - dc * w for x, w in zip(samples, window)]
    scale = .8 / max(map(abs, samples))
    path = Path(__file__).resolve().parents[1] / 'Assets/External/Church/Audio/GroundDrop.wav'
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, rate, len(samples), 'NONE', 'not compressed'))
        output.writeframes(struct.pack('<' + 'h' * len(samples), *(round(x * scale * 32767) for x in samples)))


if __name__ == '__main__':
    main()
