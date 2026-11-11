"""Sound transmission of the Church's assemblies, per octave band (125 Hz - 8 kHz).

    $PY docs/benchmarks/energy-audio-2026-11-09/church_transmission.py    # prints the table

Single leaves: Sharp's prediction (B. H. Sharp, "Prediction methods for the sound transmission of
building elements", Noise Control Engineering 11, 1978; as in Bies & Hansen, Engineering Noise
Control, 5th ed., section 7.2.4): field-incidence mass law R = 20 log10(f m) - 47 below fc/2,
R = 20 log10(f m) + 10 log10(2 eta f / (pi fc)) - 47 above the critical frequency fc, and a
straight line in log f between. fc = c^2 / (1.8 cL h) for a plate of thickness h and longitudinal
wave speed cL (glass: fc h = 12 Hz m, the usual value). Massive masonry is capped at 65 dB, the
practical limit set by flanking (Bies & Hansen, ibid.); those walls are never the leak path.
Doors are composites with their gaps (ibid. section 7.2.8): tau = (S_leaf tau_leaf + S_gap tau_gap)
/ S_total with tau_gap = 1. The Church's door leaves stop 4 cm short of the wall opening on each
side (geometry), but the door frame's stone faces cover those gaps in the acoustic geometry, so
the leak is carried by the leaf's transmission instead (no hole, no double counting): the leaf
material transmits (S_leaf tau_leaf + S_gap) / S_leaf.
Values are the transmitted share of the incident energy tau = 10^(-R/10); the material's
transmission coefficient t (shader: transmitted = (1 - a) t) is tau / (1 - a).
"""
import math

BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
C = 343.0


def sharp(m, fc, eta):
    """Sound reduction index (dB) per band of a single leaf: surface mass m (kg/m^2), critical
    frequency fc (Hz), total loss factor eta."""
    def law(f):
        return 20 * math.log10(f * m) - 47
    def above(f):
        return 20 * math.log10(f * m) + 10 * math.log10(2 * eta * f / (math.pi * fc)) - 47
    out = []
    for f in BANDS:
        if f < fc / 2:
            r = law(f)
        elif f >= fc:
            r = above(f)
        else:
            x = math.log(f / (fc / 2)) / math.log(2)
            r = law(fc / 2) + (above(fc) - law(fc / 2)) * x
        out.append(r)
    return out


def critical(cl, h):
    return C * C / (1.8 * cl * h)


# Leaves (surface mass kg/m^2, critical frequency Hz, loss factor) with their sources.
LEAVES = {
    # Leaded/stained glass: 3 mm panes (7.5 kg/m^2), fc h = 12 Hz m; lead cames damp the panes (eta 0.03).
    'glass': dict(m=7.5, fc=12.0 / .003, eta=.03, note='3 mm leaded glass panes, 7.5 kg/m2, fc 4 kHz, eta 0.03'),
    # Old solid oak door leaf, 50 mm boards: 700 kg/m^3, cL 4000 m/s along the grain, eta 0.02.
    'door leaf': dict(m=35.0, fc=critical(4000, .05), eta=.02, note='50 mm oak boards, 35 kg/m2, cL 4000 m/s, eta 0.02'),
    # Organ case and other timber: 25 mm boards.
    'timber 25 mm': dict(m=17.5, fc=critical(4000, .025), eta=.02, note='25 mm boards, 17.5 kg/m2'),
    # Pews: 16 mm boards on studs.
    'pew boards': dict(m=11.2, fc=critical(4000, .016), eta=.03, note='16 mm boards, 11.2 kg/m2'),
    # Roof: 0.6 mm copper (5.4 kg/m^2) on 25 mm softwood boarding (12.5 kg/m^2); the boarding sets
    # the bending stiffness (cL 3500 m/s), the copper adds mass and damping.
    'copper roof': dict(m=17.9, fc=critical(3500, .025), eta=.04, note='0.6 mm copper on 25 mm boarding, 17.9 kg/m2'),
    # Limestone wall 0.38 m (the thinnest wall, the west front): 2300 kg/m^3.
    'limestone wall': dict(m=874.0, fc=critical(3500, .38), eta=.02, note='0.38 m limestone, 874 kg/m2 (west front; the nave walls are 1.0-2.4 m)'),
    # Masonry vault 0.25 m.
    'masonry vault': dict(m=500.0, fc=critical(3000, .25), eta=.02, note='0.25 m masonry vault, 500 kg/m2'),
}
CAP = 65.0
# Front doors (geometry): two leaves of 7.05 m^2 each, 4.52 m tall; the opening is 4 cm wider on each side.
DOOR_LEAF_AREA = 2 * 7.05
DOOR_GAP_AREA = 2 * .04 * 4.52


def table():
    r = {k: [min(CAP, x) for x in sharp(v['m'], v['fc'], v['eta'])] for k, v in LEAVES.items()}
    leaf = [10 ** (-x / 10) for x in r['door leaf']]
    door = [(DOOR_LEAF_AREA * t + DOOR_GAP_AREA) / DOOR_LEAF_AREA for t in leaf]           # carried by the leaf area
    r['door with gaps'] = [-10 * math.log10((DOOR_LEAF_AREA * t + DOOR_GAP_AREA) / (DOOR_LEAF_AREA + DOOR_GAP_AREA)) for t in leaf]
    tau = {k: [10 ** (-x / 10) for x in v] for k, v in r.items()}
    tau['door leaf with gaps (per leaf area)'] = door
    return r, tau


# Church materials -> assembly (the shader transmission t = tau / (1 - a)).
ASSEMBLY = {'Church stone': 'limestone wall', 'Church plaster vault': 'masonry vault', 'Church floor': 'limestone wall',
            'Church wood': 'timber 25 mm', 'Church door': 'door leaf with gaps (per leaf area)', 'Church pews': 'pew boards',
            'Church glass': 'glass', 'Church copper roof': 'copper roof'}


def transmission(name, absorption):
    _, tau = table()
    return [min(1.0, t / (1 - a)) for t, a in zip(tau[ASSEMBLY[name]], absorption)]


def rw_like(r):
    """Average 125 Hz - 4 kHz sound reduction (dB), a rough single number next to published Rw."""
    return sum(r[:6]) / 6


if __name__ == '__main__':
    r, tau = table()
    print('sound reduction R (dB)      ' + ''.join(f'{f:>7d}' for f in BANDS) + '   mean 125-4k')
    for k, v in r.items():
        print(f'{k:28s}' + ''.join(f'{x:7.1f}' for x in v) + f'   {rw_like(v):5.1f}')
    print(f'door gap share of the opening: {DOOR_GAP_AREA / (DOOR_LEAF_AREA + DOOR_GAP_AREA):.1%}')
