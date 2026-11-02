"""Checks for the audio-DLSS denoiser trainer; run with unittest.

Synthetic scenes only: no dataset, Unity or GPU is needed.
"""
import unittest

import numpy as np
import torch

import dlss_baselines as bl
import dlss_train as dt


def synthetic_scene(rng, receivers=5, family='shoebox', split='train', gains=None, q=32):
    """A scene whose label is a fixed per-family mix of its first input seed's families."""
    b, s, f = len(dt.VARIANTS), 4, len(dt.FAMILIES)
    x = (rng.normal(size=(b, s, receivers, q, f)) + 1j * rng.normal(size=(b, s, receivers, q, f))).astype(np.complex64)
    gains = np.ones(f) if gains is None else np.asarray(gains)
    target = np.einsum('pqf,f->pq', x[0, 0], gains).astype(np.complex64)
    return dict(id=f'{family}-{rng.integers(1e6)}', family=family, split=split, x=x,
                counts=rng.integers(0, 9, size=x.shape).astype(np.float32),
                seconds=np.full((b, s, receivers), 1e-3), distance=rng.uniform(1, 5, receivers).astype(np.float32),
                visible=(rng.uniform(size=receivers) > .5).astype(np.float32), target=target,
                valid=np.ones((receivers, q), bool))


def inputs(scene):
    pairs = dt.as_pairs([scene])
    return [pairs[k] for k in ['x', 'counts', 'budget', 'distance', 'visible']]


def randomise(model, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.1 * torch.randn_like(p))
    return model.eval()


class Model(unittest.TestCase):
    def setUp(self):
        self.scene = synthetic_scene(np.random.default_rng(0))

    def test_gain_head_starts_at_the_raw_early_sum(self):
        for name in ['gain-k1-w32', 'gain-k3-w32', 'flat-w128']:
            model = dt.Denoiser(dt.spec_of(**dt.CANDIDATES[name])).eval()
            pred = dt.predict(model, self.scene)
            self.assertTrue(np.allclose(pred, dt.raw_prediction(self.scene), atol=1e-4), name)

    def test_output_is_scale_equivariant(self):
        for layout in dt.LAYOUTS:
            model = randomise(dt.Denoiser(dt.spec_of(layout=layout, kernel=3)))
            x, c, b, d, v = inputs(self.scene)
            with torch.no_grad():
                y = model(x, c, b, d, v)
                y2 = model(37.5 * x, c, b, d, v)
            self.assertTrue(torch.allclose(y, y2 / 37.5, rtol=1e-4, atol=1e-4), layout)

    def test_pairs_do_not_affect_each_other(self):
        model = randomise(dt.Denoiser(dt.spec_of(kernel=5, depth=3)))
        args = inputs(self.scene)
        with torch.no_grad():
            whole = model(*args)
            for i in [0, 7, len(whole) - 1]:
                alone = model(*[a[i:i + 1] for a in args])
                self.assertTrue(torch.allclose(alone[0], whole[i], rtol=1e-5, atol=1e-5))

    def test_layouts_group_families(self):
        x = torch.randn(3, 32, 8, 2)
        for layout, buffers in [('families', 8), ('no-walk', 7), ('summed-early', 2), ('summed', 1), ('slot-ranges', 4)]:
            feats = dt.Features(layout, True, 32)
            grouped = torch.einsum('nqfc,fg->nqgc', x, feats.group)
            self.assertEqual(grouped.shape[2], buffers)
        summed = torch.einsum('nqfc,fg->nqgc', x, dt.Features('summed', True, 32).group)
        self.assertTrue(torch.allclose(summed[:, :, 0], x.sum(2), atol=1e-5))
        early = torch.einsum('nqfc,fg->nqgc', x, dt.Features('summed-early', True, 32).group)
        self.assertTrue(torch.allclose(early[:, :, 0], x[:, :, :7].sum(2), atol=1e-5))
        self.assertTrue(torch.allclose(early[:, :, 1], x[:, :, 7]))
        slots = torch.einsum('nqfc,fg->nqgc', x, dt.Features('slot-ranges', True, 32).group)
        self.assertTrue(torch.allclose(slots[:, :, 2], x[:, :, 2:7].sum(2), atol=1e-5))
        self.assertTrue(torch.allclose(slots.sum(2), x.sum(2), atol=1e-5))

    def test_training_fits_a_small_fixture(self):
        rng = np.random.default_rng(3)
        gains = [1, .5, .2, 0, 0, 0, 0, 0]
        scenes = [synthetic_scene(rng, gains=gains, split=sp) for sp in ['train', 'train', 'validation']]
        for s in scenes:                          # every seed and budget sees the labelled realisation
            s['x'][:] = s['x'][0, 0]
        spec = dt.spec_of(epochs=30, batch=32, lr=1e-2, width=32)
        model, info = dt.train(spec, scenes[:2], scenes[2:], log=lambda *_: None, eval_every=30)
        before = dt.summarise([dt.scene_metrics(dt.raw_prediction(s), s) for s in scenes[:2]])
        after = dt.evaluate(model, scenes[:2])
        self.assertLess(after['all']['budgets'][dt.VARIANTS[0]]['error'],
                        .5 * before['all']['budgets'][dt.VARIANTS[0]]['error'])
        self.assertLess(info['history'][-1]['loss'], info['history'][0]['loss'])


class Data(unittest.TestCase):
    def test_pairs_weight_scenes_equally_and_shuffle_within_a_scene(self):
        rng = np.random.default_rng(1)
        scenes = [synthetic_scene(rng, receivers=7), synthetic_scene(rng, receivers=3)]
        pairs = dt.as_pairs(scenes)
        n = [len(dt.VARIANTS) * 4 * 7, len(dt.VARIANTS) * 4 * 3]
        self.assertEqual(pairs['x'].shape[0], sum(n))
        self.assertAlmostEqual(float(pairs['weight'][:n[0]].sum()), float(pairs['weight'][n[0]:].sum()), places=4)
        self.assertEqual(pairs['budget'][:n[0]].tolist(), sorted(pairs['budget'][:n[0]].tolist()))
        shuffled = dt.as_pairs(scenes, shuffle=True, seed=4)
        first = {tuple(np.round(r.numpy().ravel()[:4], 5)) for r in pairs['target'][:7]}
        moved = {tuple(np.round(r.numpy().ravel()[:4], 5)) for r in shuffled['target'][:7]}
        self.assertEqual(first, moved)              # same labels, permuted among the scene's receivers
        self.assertFalse(torch.equal(pairs['target'][:7], shuffled['target'][:7]))
        self.assertTrue(torch.equal(pairs['x'], shuffled['x']))

    def test_linear_gains_recover_a_known_mix(self):
        rng = np.random.default_rng(2)
        gains = [1, .9, .3, .1, 0, .05, .2, 0]
        scenes = [synthetic_scene(rng, gains=gains) for _ in range(3)]
        for s in scenes:
            s['x'][:] = s['x'][0, 0]
        fitted = dt.fit_linear_gains(scenes)
        self.assertTrue(np.allclose(fitted, np.broadcast_to(gains, fitted.shape), atol=1e-4))
        summed = dt.fit_linear_gains(scenes, 'summed-early')     # one early gain, one walk gain
        self.assertTrue(np.allclose(summed[:, :7], summed[:, :1]))
        self.assertTrue(np.all(summed[:, 0] < 1) and np.all(summed[:, 0] > 0))


class Metrics(unittest.TestCase):
    def test_metrics_match_the_pilot_definitions(self):
        rng = np.random.default_rng(5)
        scene = synthetic_scene(rng)
        pred = dt.raw_prediction(scene)
        row = dt.scene_metrics(pred, scene)
        for b in range(len(dt.VARIANTS)):
            ref = bl.relative_error_squared(pred[b].astype(complex), scene['target'].astype(complex))
            self.assertAlmostEqual(row['budgets'][b]['error2'], ref, places=5)
            spikes = bl.spike_scores(pred[b].astype(complex), scene['target'].astype(complex))
            self.assertAlmostEqual(row['budgets'][b]['over1'], float(np.mean(spikes > 1)), places=6)
        zero = dt.scene_metrics(np.zeros_like(pred), scene)
        self.assertAlmostEqual(zero['budgets'][0]['error2'], 1.0, places=6)
        self.assertAlmostEqual(zero['budgets'][0]['level'], 0.0)

    def test_summary_is_an_rms_over_scenes(self):
        rows = [dict(family='a', budgets=[dict(error2=e, over1=0, over3=0, level=1)] * 4) for e in (.09, .16)]
        out = dt.summarise(rows)
        self.assertAlmostEqual(out['all']['budgets'][dt.VARIANTS[0]]['error'], 100 * np.sqrt(.125), places=6)
        self.assertAlmostEqual(dt.selection_score(out), np.log(100 * np.sqrt(.125)))


if __name__ == '__main__':
    unittest.main()
