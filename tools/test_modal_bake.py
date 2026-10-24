"""Numerical checks for modal_bake; run with unittest, optional NGSolve installed."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

import modal_bake as modal


def spec(kind='plate', support='simply-supported'):
    return dict(geometry=dict(kind=kind, size=[.4, .25, .002]), support=support,
                material=dict(density=2700, young=70e9, poisson=.33, loss=.01))


class ModalChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def bank(self, config=None):
        path = self.root/'bank.npz'
        modal.bake(spec() if config is None else config, path, 12, 12000)
        return modal.read_bank(path)

    def test_plate_closed_form_frequency_and_center_selection(self):
        bank = self.bank()
        a, b, h = spec()['geometry']['size']; material = spec()['material']
        c = np.sqrt(material['young']/(12*material['density']*(1-material['poisson']**2)))
        expected = np.pi*h*c/2*(1/a**2+1/b**2)
        self.assertAlmostEqual(bank['frequencies'][0], expected)
        center = np.argmin(np.linalg.norm(bank['positions'][:, :2], axis=1))
        modes = bank['metadata']['report']['indices']
        for index, (m, n) in enumerate(modes):
            if m % 2 == 0 or n % 2 == 0:
                self.assertLess(abs(bank['shapes'][index, center, 2]), 1e-12)
        # Integrated modal mass agrees with the analytic mass normalization.
        positions, triangles = bank['positions'], bank['triangles']
        areas = np.linalg.norm(np.cross(positions[triangles[:, 1]]-positions[triangles[:, 0]],
                                       positions[triangles[:, 2]]-positions[triangles[:, 0]]), axis=1)/2
        for shape in bank['shapes']:
            mass = np.sum(areas*np.mean(shape[triangles, 2]**2, axis=1))*material['density']*h
            self.assertAlmostEqual(mass, 1, places=10)

    def test_retune_matches_fresh_plate_bake_in_frequency_and_impulse_coupling(self):
        bank = self.bank(); changed = spec(); scale = 1.7
        changed['geometry']['size'] = (np.array(changed['geometry']['size'])*scale).tolist()
        changed['material'].update(young=100e9, density=7800, loss=.03)
        modal.retune(bank, self.root/'tuned.npz', young=100e9, density=7800, loss=.03, scale=scale)
        modal.bake(changed, self.root/'fresh.npz', 12, 12000)
        tuned, fresh = modal.read_bank(self.root/'tuned.npz'), modal.read_bank(self.root/'fresh.npz')
        for key in ('frequencies', 'positions', 'shapes'):
            assert_allclose(tuned[key], fresh[key], atol=1e-12, rtol=1e-12)

    def test_rotated_box_fit_recovers_dimensions_and_volume(self):
        config = spec('box', 'free')
        config['geometry'].update(size=[2, 1, .4], center=[10, -4, 8],
                                  basis=Rotation.from_rotvec([.2, .5, .4]).as_matrix().tolist())
        vertices, faces = modal.box_surface(config['geometry'])
        config['geometry'] = dict(kind='mesh', vertices=vertices, triangles=faces)
        fit = modal.fit_box(config)
        assert_allclose(fit['geometry']['size'], [2, 1, .4], atol=1e-12)
        self.assertAlmostEqual(fit['fit']['volumeFill'], 1)
        self.assertLess(fit['fit']['surfaceRmsMeters'], 1e-12)

    def test_box_fit_is_stable_under_render_vertex_duplication(self):
        config = spec('box', 'free'); vertices, faces = modal.box_surface(config['geometry'])
        reference = modal.fit_box(config)
        config['geometry'] = dict(kind='mesh', vertices=vertices[faces].reshape(-1, 3),
                                  triangles=np.arange(faces.size).reshape(-1, 3))
        actual = modal.fit_box(config)
        assert_allclose(actual['geometry']['size'], reference['geometry']['size'])
        self.assertLess(actual['fit']['surfaceRmsMeters'], 1e-12)

    def test_nonbox_fit_exposes_volume_error(self):
        config = spec('mesh', 'free')
        config['geometry'].update(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  triangles=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        fit = modal.fit_box(config)
        self.assertGreater(fit['fit']['volumeRelativeError'], 1)
        self.assertGreater(fit['fit']['surfaceRmsMeters'], .01)
        self.assertIn('unmeasured', fit['fit']['acousticError'])

    def test_invalid_or_disconnected_surfaces_are_rejected(self):
        vertices, faces = modal.box_surface(spec('box')['geometry'])
        config = dict(kind='mesh', vertices=vertices, triangles=faces[:-1])
        with self.assertRaises(ValueError): modal.solid_surface(config)
        config.update(vertices=np.vstack((vertices, vertices+[2, 0, 0])),
                      triangles=np.vstack((faces, faces+8)))
        with self.assertRaises(ValueError): modal.solid_surface(config)
        config.update(vertices=vertices, triangles=faces[:, ::-1])
        _, _, volume = modal.solid_surface(config)
        self.assertAlmostEqual(volume, .4*.25*.002)

    def test_batch_partitions_preserve_shared_resonances_and_tails(self):
        bank = self.bank()
        node = np.argmax(abs(bank['shapes'][0, :, 2])); triangle = int(np.where(bank['triangles'] == node)[0][0])
        weights = (bank['triangles'][triangle] == node).astype(float).tolist()
        point = dict(triangle=triangle, barycentric=weights, direction=[0, 0, 1])
        observation = modal.coupling(bank, **point)
        impacts = [dict(point, sample=sample, impulse=impulse) for sample, impulse in
                   [(0, .001), (17, .002), (17, -.0004), (129, .001), (510, -.0002)]]
        reference = modal.Resonator(bank, 48000).render(3000, impacts, observation)
        for size in (1, 17, 128, 1000):
            state = modal.Resonator(bank, 48000); blocks = []
            while state.sample < 3000:
                count = min(size, 3000-state.sample)
                events = [e for e in impacts if state.sample <= e['sample'] < state.sample+count]
                blocks.append(state.render(count, events, observation))
            assert_allclose(np.concatenate(blocks), reference, rtol=0, atol=0)
        self.assertGreater(np.linalg.norm(reference[511:]), 0)

    def test_impulse_solution_matches_closed_form_and_superposition(self):
        bank = self.bank(); observation = np.ones(len(bank['frequencies']))
        event = dict(sample=7, impulse=.001, triangle=100, barycentric=[.2, .3, .5], direction=[0, 0, 1])
        state = modal.Resonator(bank, 48000)
        output = state.render(1000, [event], observation)
        time = np.maximum(0, np.arange(1000)-7)/48000
        drive = .001*modal.coupling(bank, 100, [.2, .3, .5], [0, 0, 1])
        envelope = np.exp(-time[:, None]*state.alpha)
        expected = envelope*np.sin(time[:, None]*state.beta)/state.beta*drive
        expected[:7] = 0
        assert_allclose(output[:, 0], expected.sum(axis=1), atol=1e-15, rtol=1e-10)
        doubled = modal.Resonator(bank, 48000).render(1000, [event, event], observation)
        assert_allclose(doubled, 2*output, atol=0, rtol=0)

    def test_misconfigured_plate_material_and_timestamp_fail(self):
        config = spec(); config['geometry']['size'][2] = .1
        with self.assertRaises(ValueError): self.bank(config)
        config = spec(); config['material']['poisson'] = .5
        with self.assertRaises(ValueError): self.bank(config)
        bank = self.bank(); state = modal.Resonator(bank, 48000)
        with self.assertRaises(ValueError):
            state.render(10, [dict(sample=10)], np.ones(len(bank['frequencies'])))
        with self.assertRaises(ValueError): modal.Resonator(bank, 100)

    @unittest.skipUnless(importlib.util.find_spec('ngsolve'), 'NGSolve optional')
    def test_solid_fem_nullspace_refinement_and_scaling(self):
        config = spec('box', 'free'); config['geometry']['size'] = [.4, .25, .1]
        modal.bake(config, self.root/'coarse.npz', 8, 50000, .055, 2)
        modal.bake(config, self.root/'fine.npz', 8, 50000, .035, 2)
        coarse, fine = modal.read_bank(self.root/'coarse.npz'), modal.read_bank(self.root/'fine.npz')
        self.assertEqual(fine['metadata']['report']['rigidModesRemoved'], 6)
        self.assertLess(np.max(abs(coarse['frequencies']/fine['frequencies']-1)), .03)
        changed = copy.deepcopy(config); changed['material'].update(young=100e9, density=7800)
        modal.bake(changed, self.root/'changed.npz', 8, 50000, .035, 2)
        modal.retune(fine, self.root/'retuned.npz', young=100e9, density=7800)
        assert_allclose(modal.read_bank(self.root/'changed.npz')['frequencies'],
                        modal.read_bank(self.root/'retuned.npz')['frequencies'], rtol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec('ngsolve'), 'NGSolve optional')
    def test_clamped_solid_beam_agrees_with_slender_beam_theory(self):
        config = spec('box', 'clamp-x-min')
        length, width, height = .6, .03, .02
        config['geometry']['size'] = [length, width, height]
        modal.bake(config, self.root/'beam.npz', 2, 1000, .025, 2)
        actual = modal.read_bank(self.root/'beam.npz')['frequencies']
        mat = config['material']
        # First bending mode in each plane of a uniform Euler–Bernoulli cantilever.
        expected = 1.875104068711961**2/(2*np.pi*length**2) * np.sqrt(
            mat['young']/mat['density'])*np.array([height, width])/np.sqrt(12)
        assert_allclose(actual, expected, rtol=.015)

    @unittest.skipUnless(importlib.util.find_spec('ngsolve'), 'NGSolve optional')
    def test_closed_mesh_bake_and_supports(self):
        config = spec('box', 'free'); config['geometry']['size'] = [.4, .25, .1]
        vertices, triangles = modal.box_surface(config['geometry'])
        config['geometry'] = dict(kind='mesh', vertices=vertices.tolist(), triangles=triangles.tolist())
        modal.bake(config, self.root/'mesh.npz', 6, 50000, .09, 2)
        mesh = modal.read_bank(self.root/'mesh.npz')
        self.assertLess(mesh['metadata']['report']['relativeResidualMax'], 1e-7)
        config['support'] = 'clamp-x-min'
        modal.bake(config, self.root/'clamped.npz', 6, 50000, .09, 2)
        fixed = modal.read_bank(self.root/'clamped.npz')
        self.assertEqual(fixed['metadata']['report']['rigidModesRemoved'], 0)
        face = abs(fixed['positions'][:, 0]+.2) < 1e-8
        assert_allclose(fixed['shapes'][:, face], 0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
