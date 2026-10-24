"""New public inference routing, tested on CPU without constructing a model."""
import importlib.util
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

class InferenceModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('public_prediction', ROOT / 'src/predict.py')
        cls.prediction = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.prediction)

    def test_mode_selection_does_not_need_private_submission_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'configs').mkdir()
            for name, views in [('single', ['odd']), ('dual', ['odd', 'even'])]:
                (root / 'configs' / ('inference_' + name + '.json')).write_text(json.dumps(dict(
                    views=views, batch_size=8, virtual_reference_ranks=4)))
            with patch.object(self.prediction, 'ROOT', root):
                self.assertEqual(self.prediction.select_config(None)['views'], ['odd', 'even'])
                self.assertEqual(self.prediction.select_config('dual')['views'], ['odd', 'even'])
                self.assertEqual(self.prediction.select_config('single')['views'], ['odd'])
                self.assertEqual(self.prediction.select_config('56156565')['views'], ['odd', 'even'])
                self.assertEqual(self.prediction.select_config('56125089')['views'], ['odd'])
                with self.assertRaises(ValueError):
                    self.prediction.select_config('unknown')

    def test_cli_help_exposes_explicit_research_checkpoint(self):
        result = subprocess.run([sys.executable, str(ROOT / 'src/predict.py'), '--help'],
                                text=True, capture_output=True, timeout=15,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--checkpoint', result.stdout)
        self.assertIn('--expected-sha256', result.stdout)

    def test_explicit_checkpoint_integrity_and_default_frozen_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model.pt'
            model.write_bytes(b'synthetic-file-for-integrity-only')
            args = argparse.Namespace(checkpoint=str(model), expected_sha256=None)
            path, digest = self.prediction.resolve_checkpoint(args, {})
            self.assertEqual(path, model.resolve())
            self.assertEqual(digest, self.prediction.sha256(model))
            args.expected_sha256 = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                self.prediction.resolve_checkpoint(args, {})
            with patch.object(self.prediction, 'ROOT', root):
                args = argparse.Namespace(checkpoint=None, expected_sha256=None)
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    self.prediction.resolve_checkpoint(args, {'checkpoint': 'model.pt'})
            self.assertEqual(model.read_bytes(), b'synthetic-file-for-integrity-only')

    def test_help_does_not_import_the_ml_stack(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('cv2', sys.modules)

    def test_inference_dispatch_retains_inference_mode_context(self):
        events = []
        @contextmanager
        def mode():
            events.append('enter')
            yield
            events.append('exit')
        def body(args, work):
            self.assertEqual(events, ['enter'])
            return 'dispatched'
        with patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=mode)}):
            with patch.object(self.prediction, '_run', body):
                self.assertEqual(self.prediction.run(None, None), 'dispatched')
        self.assertEqual(events, ['enter', 'exit'])

    def test_arbitrary_label_free_input_population_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data/small_model_track_test'
            data.mkdir(parents=True)
            (root / 'test.csv').write_text('path\nsmall_model_track_test/clip-a/\nsmall_model_track_test/clip-b/\n')
            source, manifest = self.prediction.locate_input(root)
            self.assertEqual(source, (root / 'data').resolve())
            self.assertEqual(manifest, (root / 'test.csv').resolve())
            manifest.write_text('path\nsmall_model_track_test/../\n')
            with self.assertRaises(ValueError):
                self.prediction.locate_input(root)


if __name__ == '__main__':
    unittest.main()
