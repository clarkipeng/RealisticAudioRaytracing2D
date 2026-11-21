"""The train station's acoustic materials (docs/benchmarks/path-denoise-2026-11-12, part 43).

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/station_acoustics.py inspect   # guarded Unity: the take's geometry and its visual materials
    $PY tools/station_acoustics.py plan                  # materials and rules -> .context/station-acoustics/plan.json
    nice -n 19 $PY tools/station_acoustics.py apply     # guarded Unity: material assets and assignments; copied back

Every surface of the station used the take's default material (Assets/Materials/Material.asset). Here each kind of
surface gets published octave-band data, assigned by the visual material its triangles carry (StationAcoustics.cs):

Absorption (125 Hz - 8 kHz):
- M. Vorlaender, Auralization (Springer 2008), annex tables, as transcribed in the pyroomacoustics material database
  (keys in brackets), as for the Church (docs/benchmarks/energy-audio-2026-11-09, "Materials");
- the classic table reproduced by sengpielaudio (https://sengpielaudio.com/calculator-RT60Coeff.htm; 8 kHz repeats
  4 kHz, as for the Church);
- railway ballast: no octave table was found in print; its absorption is computed from the ground impedance the
  railway-noise literature fits to ballast: a semi-infinite porous ground of flow resistivity 50 kPa s/m^2 (X. Zhang,
  D. J. Thompson et al., "An engineering model for the sound radiation from railway track", ISVR Southampton, eprints
  433604, which uses it with the Delany-Bazley model; Broadbent et al. fitted 25-51 kPa s/m^2 to reverberation-room
  measurements), with Miki's form of the model (Y. Miki, J. Acoust. Soc. Jpn (E) 11, 1990: physically admissible where
  Delany-Bazley is not), locally reacting, averaged over random incidence (Paris) and over the third octaves of each band.
- grass: the same model with the ground class's flow resistivity;
- painted steel: no octave table either; the hardest sealed surface tabulated, reverberation-chamber walls
  [reverb_chamber], stands for it (a smooth painted steel skin absorbs no more than that; its panel resonances are in
  its transmission, below).

Scattering (ISO 17497-1 sense, random incidence): Kirchhoff's theory of a randomly rough surface (P. Beckmann &
A. Spizzichino, The Scattering of Electromagnetic Waves from Rough Surfaces, 1963; J. A. Ogilvy, Theory of Wave
Scattering from Random Rough Surfaces, 1991) keeps exp(-(2 k h cos theta)^2) of the energy specular, h the rms
height; over random incidence (weight 2 cos theta sin theta) the scattered share is s = 1 - (1 - exp(-a)) / a,
a = (2 k h)^2. Each surface's h is stated with its source; masonry keeps the Church's calibrated stone masonry
values (Planells et al. 2016). Shapes larger than the roughness (the canopy's ribs, the train's features, the
benches) are geometry, not scattering.

Transmission: Sharp's prediction of the sound reduction of single leaves (church_transmission.py), per assembly
below; tau = 10^(-R/10), the shader's t = tau / (1 - a).
"""
import cmath
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'docs/benchmarks/energy-audio-2026-11-09'))
import church_transmission as ct  # noqa: E402

WORK = ROOT / '.context/station-acoustics'
SCENE = 'Assets/Scenes/TrainStation.unity'
FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000]
C, RHO = 343.0, 1.21


def miki_ground(sigma):
    """Random-incidence absorption per octave band of a locally reacting semi-infinite ground of flow resistivity
    sigma (Pa s/m^2), Miki's model, averaged over the band's three third octaves."""
    def alpha(f):
        x = f / sigma
        zc = RHO * C * complex(1 + .0699 * x ** -.632, -.1071 * x ** -.632)
        total, n = 0.0, 400
        for i in range(n):   # Paris: alpha_random = int_0^(pi/2) alpha(theta) sin(2 theta) d theta (midpoint rule)
            theta = (i + .5) * (math.pi / 2) / n
            r = (zc * math.cos(theta) - RHO * C) / (zc * math.cos(theta) + RHO * C)
            total += (1 - abs(r) ** 2) * math.sin(2 * theta) * (math.pi / 2) / n
        return total
    return [sum(alpha(f * 2 ** (k / 3)) for k in (-1, 0, 1)) / 3 for f in FREQUENCIES]


def kirchhoff(h):
    """Random-incidence scattering coefficient per band of a surface of rms height h (m)."""
    out = []
    for f in FREQUENCIES:
        a = (2 * 2 * math.pi * f / C * h) ** 2
        out.append(0.0 if a == 0 else 1 - (1 - math.exp(-a)) / a)
    return out


def steel_leaf(h, eta):
    """Sharp's leaf of steel h (m) thick: 7850 kg/m^3, longitudinal wave speed 5200 m/s."""
    return dict(m=7850 * h, fc=ct.critical(5200, h), eta=eta)


# Assemblies: Sharp's single leaves, or None for massive construction (capped at 65 dB, as the Church's walls).
ASSEMBLIES = {
    'massive': None,
    # Canopy and waiting-room glazing: 10 mm toughened glass, 25 kg/m^2, fc h = 12 Hz m, eta 0.01 (a mounted pane).
    'glass 10 mm': dict(m=25.0, fc=12.0 / .010, eta=.01),
    # Train body: a 2 mm steel skin (the leaf; the lining behind it is not modelled), eta 0.01.
    'steel 2 mm': steel_leaf(.002, .01),
    # Canopy roof sheet and fittings: 0.7 mm profiled steel sheet, eta 0.01.
    'steel sheet 0.7 mm': steel_leaf(.0007, .01),
    # Bench slats: the Church's 25 mm timber boards.
    'timber 25 mm': ct.LEAVES['timber 25 mm'],
}


def transmission(assembly, absorption):
    leaf = ASSEMBLIES[assembly]
    r = [ct.CAP] * 7 if leaf is None else [min(ct.CAP, x) for x in ct.sharp(leaf['m'], leaf['fc'], leaf['eta'])]
    t = [min(1.0, 10 ** (-x / 10) / (1 - a)) for x, a in zip(r, absorption)]
    return t + [t[-1]]   # 16 kHz = 8 kHz


SENG_BRICK = [.03, .03, .03, .04, .05, .07, .07]            # sengpielaudio: Brick, unglazed (8 kHz = 4 kHz)
SENG_PLATE_GLASS = [.18, .06, .04, .03, .02, .02, .02]       # sengpielaudio: Glass, 1/4" plate, large (8 kHz = 4 kHz)
PLANELLS_MASONRY = [.12, .13, .14, .15, .16, .17, .17]       # Planells et al. 2016, stone masonry (the Church's)

MATERIALS = {
    'Station concrete': dict(
        absorption=[.02, .03, .03, .03, .04, .07, .07], scattering=kirchhoff(.001), assembly='massive',
        note='Platform, boarding deck and steps, canopy slab: rough concrete [rough_concrete]; scattering: Kirchhoff, rms '
             'height 1 mm (brushed concrete, macrotexture depth about 1 mm, EN 13036-1); massive'),
    'Station brick': dict(
        absorption=SENG_BRICK, scattering=PLANELLS_MASONRY, assembly='massive',
        note="Waiting room walls: unglazed brick (sengpielaudio); scattering: stone masonry (Planells et al. 2016, the Church's); massive"),
    'Station steel': dict(
        absorption=[.01, .01, .01, .02, .02, .04, .04], scattering=kirchhoff(0), assembly='steel sheet 0.7 mm',
        note='Waiting room roof, columns, lamp posts, rails, speaker housing: painted steel, as the hardest tabulated sealed '
             'surface [reverb_chamber]; smooth; 0.7 mm steel sheet (Sharp)'),
    'Station steel panel': dict(
        absorption=[.01, .01, .01, .02, .02, .04, .04], scattering=kirchhoff(0), assembly='steel 2 mm',
        note='Train body and cab, carriage sides, ends and roof, the sliding door: painted steel [reverb_chamber]; smooth; '
             '2 mm steel skin (Sharp)'),
    'Station timber': dict(
        absorption=[.18, .12, .10, .09, .08, .07, .07], scattering=kirchhoff(.001), assembly='timber 25 mm',
        note="Benches, the carriage's floor: 16 mm wood on 40 mm studs [wood_16mm] (as the Church pews); Kirchhoff, rms "
             'height 1 mm (planed boards); 25 mm boards (Sharp)'),
    'Station ballast': dict(
        absorption=None, scattering=kirchhoff(.010), assembly='massive', sigma=50e3,
        note='Track bed: railway ballast, Miki ground of 50 kPa s/m2 (Zhang, Thompson et al.; Broadbent et al. 25-51), '
             'random incidence; Kirchhoff, rms height 10 mm (an estimate: a quarter of the 40 mm mean stone of the EN 13450 '
             '31.5-50 mm grading); massive'),
    'Station ground': dict(
        absorption=None, scattering=kirchhoff(0), assembly='massive', sigma=200e3,
        note='Terrain (grass): Miki ground of 200 kPa s/m2, the Nord2000 class D "normal uncompacted ground: pasture, forest '
             'floor" (Plovsing 2006; Embleton, Piercy & Daigle, JASA 74, 1983, measured 150-300 for grass), random incidence; '
             'smooth (its roughness is not modelled); massive'),
}
for _m in MATERIALS.values():
    if _m['absorption'] is None:
        _m['absorption'] = [round(x, 3) for x in miki_ground(_m['sigma'])]


def plan(rules):
    """The plan: materials, and the rules mapping visual materials (and object names) to them."""
    data = dict(
        materials=[dict(name=n, note=m['note'], frequencies=FREQUENCIES, absorption=[round(x, 4) for x in m['absorption']],
                        scattering=[round(x, 4) for x in m['scattering']], transmission=transmission(m['assembly'], m['absorption']),
                        # Thin leaves modelled as closed boxes are partitions (index 1: their reduction applies once).
                        ior=[1.0] * 8 if m['assembly'] != 'massive' else [])
                   for n, m in MATERIALS.items()],
        visuals=[dict(visual=v, material=m) for v, m in rules['visuals'].items()],
        objects=[dict(visual=o, material=m) for o, m in rules['objects'].items()],
        fallback=rules['fallback'], trackBed='Station ballast')
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'plan.json').write_text(json.dumps(data, indent=1) + '\n')
    for m in data['materials']:
        print(f"{m['name']:18s} a {' '.join(f'{x:.3f}' for x in m['absorption'])}  s {' '.join(f'{x:.2f}' for x in m['scattering'])}  "
              f"t {' '.join(f'{x:.1e}' for x in m['transmission'][:7])}")
    print(f'-> {WORK / "plan.json"}')


# Visual material -> acoustic material (StationAcoustics.Apply), and objects whose name decides instead.
RULES = dict(
    visuals={'Concrete': 'Station concrete', 'Masonry': 'Station brick', 'Steel': 'Station steel',
             'Wayfinding': 'Station steel panel', 'Timber': 'Station timber', 'Ground': 'Station ground'},
    objects={'Canopy': 'Station concrete', 'Open carriage roof': 'Station steel panel',
             'Body acoustic proxy': 'Station steel panel', 'Cab acoustic proxy': 'Station steel panel'},
    fallback='Station concrete')


def unity(name, method, *args):
    import method_runs as mr
    import dlss_jobs as dj
    WORK.mkdir(parents=True, exist_ok=True)
    mr.LOGS = WORK / 'logs'; mr.LOGS.mkdir(parents=True, exist_ok=True)
    for rel in [SCENE, SCENE + '.meta'] + [str(p.relative_to(ROOT)) for p in (ROOT / 'Assets/Materials').glob('Station *.asset*')]:
        target = dj.RUNNER / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    logs, resources = mr.launch(name, ['-quit', '-executeMethod', method, *args])
    if resources['exitCode'] != 0:
        raise SystemExit(f'{method} failed; see {logs}/unity.log')
    return dj.RUNNER


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'inspect':
        unity('inspect', 'StationAcoustics.Inspect', '-stationOutput', str(WORK))
        inventory = json.loads((WORK / 'inventory.json').read_text())
        from collections import Counter
        print(Counter(tuple(g['visual']) for g in inventory['geometry']).most_common())
    elif command == 'plan':
        plan(RULES)
    elif command == 'apply':
        runner = unity('apply', 'StationAcoustics.Apply', '-stationPlan', str(WORK / 'plan.json'))
        for path in [SCENE] + [str(p.relative_to(runner)) for p in (runner / 'Assets/Materials').glob('Station *.asset*')]:
            shutil.copy2(runner / path, ROOT / path)
        print('applied; scene and Station materials copied back')
    else:
        raise SystemExit(__doc__)
