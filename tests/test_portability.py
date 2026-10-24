"""Tests for new public-data/config adapters; no ML dependencies or real data."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = load_script('prepare_manifest')
run = load_script('prepare_run')
fetch = load_script('fetch_pretrained')


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.folds = json.loads((ROOT / 'configs/subject_folds.json').read_text())['folds']

    def test_manifest_preserves_encounter_order_and_natural_absence(self):
        names = ['HAR/data/IR/6_Drink_water/user4/1-1-1/1.png',
                 'HAR/data/Thermal/6_Drink_water/user4/1-1-1/1.png',
                 'HAR/data/Radar/7_Eat_food/user2/1-1-2/points.csv',
                 'HAR/data/IR/6_Drink_water/user4/1-1-1/2.png',
                 '__MACOSX/._ignored',
                 'HAR/data/Radar/8_Example/user9/1-1-1/visualizations/ignored.png']
        rows = manifest.manifest_rows(names, self.folds)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['action_id'] for r in rows], [6, 7])
        self.assertEqual([r['validation_fold'] for r in rows], [2, 1])
        self.assertEqual(rows[0]['IR_file_count'], 2)
        self.assertEqual(rows[0]['IMU_path'], '')
        self.assertEqual(rows[1]['Radar_path'], 'HAR/data/Radar/7_Eat_food/user2/1-1-2')

    def test_unexpected_user_rejected(self):
        with self.assertRaisesRegex(ValueError, 'subject'):
            manifest.manifest_rows(['HAR/data/IR/0_Example/user99/t/1.png'], self.folds)

    def test_fold_overlap_rejected(self):
        bad = copy.deepcopy(self.folds)
        bad['2']['subjects'].append(2)
        with self.assertRaisesRegex(ValueError, 'multiple folds'):
            manifest.manifest_rows([], bad)

    def test_label_outside_competition_rejected(self):
        with self.assertRaisesRegex(ValueError, 'action'):
            manifest.manifest_rows(['HAR/data/IR/40_Invalid/user1/t/1.png'], self.folds)

    def test_partial_zip_cli_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive, output = base / 'toy.zip', base / 'manifest.csv'
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('HAR/data/IR/0_Example/user1/t/1.png', b'synthetic-header-only')
            command = [sys.executable, str(ROOT / 'scripts/prepare_manifest.py'),
                       '--source', str(archive), '--out', str(output)]
            rejected = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(output.exists())
            accepted = subprocess.run(command + ['--allow-partial'], text=True, capture_output=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            original = output.read_bytes()
            repeated = subprocess.run(command + ['--allow-partial'], text=True, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


class TrainingAdapterTests(unittest.TestCase):
    def test_imu_full_and_fold_stops(self):
        full, fold = run.imu_recipe(0), run.imu_recipe(1)
        self.assertEqual(full['epochs'], 43)
        self.assertTrue(full['full_source_fit'])
        self.assertEqual(fold['epochs'], 50)
        self.assertFalse(fold['full_source_fit'])
        self.assertEqual(full['schedule_epochs'], 50)
        self.assertTrue(Path(full['pretrained']).is_absolute())

    def source_fixture(self, base, fold=1):
        cache, imu = base / 'cache', base / 'imu'
        cache.mkdir(); imu.mkdir()
        records = [dict(subject=2, fold=1), dict(subject=1, fold=2), dict(subject=5, fold=3)] * 1012
        meta = dict(split='train', signature='fixture-cache-signature', records=records)
        source = dict(fold=fold, cache_signature=meta['signature'], uses_test_data=False,
                      synthetic_smoke=False, git_dirty=False, git_commit='a' * 40,
                      train_subjects=[1, 5] if fold else [1, 2, 5],
                      validation_subjects=[2] if fold else [],
                      config=dict(pretrained_sha256=run.PUBLIC_IMU_SHA))
        if fold == 0:
            source['full_source_fit'] = True
        (cache / 'cache.json').write_text(json.dumps(meta))
        (imu / 'run.json').write_text(json.dumps(source))
        (imu / 'complete.json').write_text('{}')
        (imu / 'best.pt').write_bytes(b'synthetic-handoff-file-not-a-model')
        return cache, imu, source

    def test_joint_handoff_uses_actual_public_run_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, imu, source = self.source_fixture(Path(temporary))
            config = run.joint_recipe(1, cache, imu)
            identity = config['source_sensor_initialization']
            self.assertEqual(identity['source_git_commit'], 'a' * 40)
            self.assertEqual(identity['cache_signature'], 'fixture-cache-signature')
            self.assertEqual(identity['folds']['1']['sha256'], run.file_hash(imu / 'best.pt'))
            self.assertNotIn('full_source_fit', identity)
            self.assertEqual(config['epochs'], 50)

    def test_full_source_initializer_rejected_for_held_user_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, imu, source = self.source_fixture(Path(temporary))
            source['full_source_fit'] = True
            (imu / 'run.json').write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'Full-source'):
                run.joint_recipe(1, cache, imu)

    def test_wrong_source_users_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, imu, source = self.source_fixture(Path(temporary))
            source['train_subjects'] = [1, 2, 5]
            (imu / 'run.json').write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'users'):
                run.joint_recipe(1, cache, imu)

    def test_full_data_recipe_preserves_fixed_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, imu, source = self.source_fixture(Path(temporary), fold=0)
            config = run.joint_recipe(0, cache, imu)
            self.assertEqual(config['epochs'], 20)
            self.assertEqual(config['schedule_epochs'], 50)
            self.assertTrue(config['source_sensor_initialization']['full_source_fit'])


class AssetTests(unittest.TestCase):
    def test_wrong_existing_asset_is_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory / fetch.ASSETS['imu'][0]
            target.write_bytes(b'not-the-pinned-public-checkpoint')
            with self.assertRaisesRegex(ValueError, 'preserved'):
                fetch.download('imu', directory)
            self.assertEqual(target.read_bytes(), b'not-the-pinned-public-checkpoint')


if __name__ == '__main__':
    unittest.main()
