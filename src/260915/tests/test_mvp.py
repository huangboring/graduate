"""CPU contract tests; these do not validate the CUDA model or pose accuracy."""
import ast
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
import cv2
import numpy as np
import torch
from mvp_demo.selection import select, select_payload, describe, QueryKeySelector
from mvp_demo.source import inspect_sequence, PanopticVideos, prepare_frame, open_source, annotation_files


def camera(name='00_00'):
    return dict(name=name, panel=0, node=int(name[-2:]), resolution=[64, 48],
                K=[[60.,0.,32.],[0.,60.,24.],[0.,0.,1.]], R=np.eye(3).tolist(),
                t=[10.,20.,100.], distCoef=[0.] * 5)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ids = ['00_00', '00_01', '00_02']
        self.features = [torch.stack([torch.full((8,4,4), float(v)) for v in (1,5,2)])]

    def test_all_has_zero_savings(self):
        result = select(self.features, self.ids, 3)
        self.assertEqual(result.indices, [0,1,2])
        self.assertEqual(result.total_bytes, self.features[0].numel()*4)
        self.assertEqual(result.saving_fraction, 0)

    def test_fixed_keeps_metadata_aligned(self):
        result = select(self.features, self.ids, 2, 'fixed', ['00_02','00_00'])
        features, images, meta = select_payload(self.features, self.ids, [10,11,12], result)
        self.assertEqual(images, ['00_00','00_02'])
        self.assertEqual(meta, [10,12])
        torch.testing.assert_close(features[0], self.features[0][[0,2]])

    def test_dynamic_uses_small_message_and_counts_overhead(self):
        result = select(self.features, self.ids, 2, 'descriptor')
        self.assertEqual(result.indices, [1,2])
        self.assertEqual(result.descriptor_bytes, 3*32*4)
        self.assertEqual(result.total_bytes, 2*8*4*4*4 + 3*32*4 + 3)
        changed = [self.features[0].clone()]
        changed[0][0] = 10
        self.assertEqual(select(changed, self.ids, 2, 'descriptor').indices, [0,1])

    def test_learned_has_no_silent_random_fallback(self):
        with self.assertRaises(ValueError):
            select(self.features, self.ids, 2, 'learned')
        scorer = QueryKeySelector()
        loss = scorer(describe(self.features)).sum()
        loss.backward()
        self.assertIsNotNone(scorer.query.weight.grad)

    def test_reject_invalid_inputs(self):
        for k in (1,4):
            with self.assertRaises(ValueError):
                select(self.features, self.ids, k)
        with self.assertRaises(ValueError):
            select(self.features, self.ids, 2, 'fixed', ['00_00','00_00'])
        with self.assertRaises(ValueError):
            select([torch.zeros(6,8,4,4)], self.ids, 2)


class SourceTests(unittest.TestCase):
    def test_legacy_images_preserve_sparse_hd_frame_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence=Path(directory)/'legacy';sequence.mkdir()
            (sequence/'calibration_legacy.json').write_text(json.dumps({'cameras':[camera('00_00'),camera('00_01')]}))
            for name in ('00_00','00_01'):
                folder=sequence/'hdVideos'/f'hd_{name}';folder.mkdir(parents=True)
                for index in (173,175,180):
                    cv2.imwrite(str(folder/f'{index:08d}.jpg'),np.full((48,64,3),index%255,np.uint8))
            ids,paths,_=inspect_sequence(sequence,input_format='legacy_images')
            source=open_source(paths,start=173)
            self.assertEqual(ids,['00_00','00_01'])
            self.assertEqual(source.read()[0],173)
            source.skip(2)
            self.assertEqual(source.read()[0],180)
            self.assertIsNone(source.read())
            source.close()
            (paths[0]/'00000175.jpg').unlink()
            with self.assertRaises(ValueError):open_source(paths)

    def test_official_images_and_nested_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence=Path(directory)/'official';sequence.mkdir()
            (sequence/'calibration_official.json').write_text(json.dumps({'cameras':[camera('00_00'),camera('00_01')]}))
            for name in ('00_00','00_01'):
                folder=sequence/'hdImgs'/name;folder.mkdir(parents=True)
                cv2.imwrite(str(folder/f'{name}_00000173.jpg'),np.zeros((48,64,3),np.uint8))
            _,paths,_=inspect_sequence(sequence)
            self.assertEqual(open_source(paths,start=0).read()[0],173)
            nested=sequence/'hdPose3d_stage1_coco19'/'hdPose3d_stage1_coco19';nested.mkdir(parents=True)
            (nested/'body3DScene_00000173.json').write_text('{"bodies":[]}')
            self.assertEqual(list(annotation_files(sequence)),[173])
            (nested.parent/'body3DScene_00000173.json').write_text('{"bodies":[]}')
            with self.assertRaises(ValueError):annotation_files(sequence)

    def test_calibration_matches_panoptic_projection(self):
        c = camera()
        image, meta = prepare_frame(np.zeros((48,64,3), np.uint8), c, [64,48], [16,12], 'cpu')
        self.assertEqual(tuple(image.shape), (1,3,48,64))
        self.assertNotIn('joints_3d', meta)
        world_mm = np.array([20.,30.,40.])
        axis = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        original_cm = axis @ world_mm / 10.
        expected_cm = np.asarray(c['R']) @ original_cm + np.asarray(c['t'])
        cam = meta['camera']
        actual_mm = cam['R'][0].numpy() @ (world_mm-cam['T'][0,:,0].numpy())
        np.testing.assert_allclose(actual_mm / 10, expected_cm)
        np.testing.assert_allclose(meta['affine_trans'][0] @ meta['inv_affine_trans'][0], np.eye(3), atol=1e-5)

    def test_reject_resolution_mismatch(self):
        with self.assertRaises(ValueError):
            prepare_frame(np.zeros((24,32,3), np.uint8), camera(), [64,48], [16,12], 'cpu')

    def test_video_discovery_shortest_stream_and_seek(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory) / 'example'
            videos = sequence / 'hdVideos'
            videos.mkdir(parents=True)
            (sequence / 'calibration_example.json').write_text(json.dumps({'cameras':[camera('00_00'),camera('00_01')]}))
            for name, length in [('00_00',4),('00_01',3)]:
                writer = cv2.VideoWriter(str(videos / f'hd_{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30., (64,48))
                self.assertTrue(writer.isOpened())
                for frame in range(length):
                    writer.write(np.full((48,64,3), frame*50, dtype=np.uint8))
                writer.release()
            ids, paths, _ = inspect_sequence(sequence)
            self.assertEqual(ids, ['00_00','00_01'])
            with self.assertRaises(ValueError):
                inspect_sequence(sequence, ['00_00','00_02'])
            source = PanopticVideos(paths, start=1)
            try:
                frame, images = source.read()
                self.assertEqual(frame, 1)
                self.assertLess(abs(float(images[0].mean()) - 50), 8)
                self.assertEqual(source.read()[0], 2)
                self.assertIsNone(source.read())
            finally:
                source.close()


class ModelContractTests(unittest.TestCase):
    @staticmethod
    def methods(*names):
        # Exercise actual modified methods without importing unrelated CUDA dependencies.
        tree = ast.parse((ROOT/'lib/models/dq_transformer.py').read_text(encoding='utf-8'))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'DyanmicQueryTransformer')
        module = ast.Module(body=[x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names], type_ignores=[])
        scope = dict(torch=torch, math=math)
        exec(compile(ast.fix_missing_locations(module), '<actual model methods>', 'exec'), scope)
        return scope

    def test_sample_space_without_any_ground_truth(self):
        scope = self.methods('initialize_reference_points')
        dummy = types.SimpleNamespace(num_joints=15, norm2absolute=lambda x:x,
            generate_T_pose=lambda roots:roots[:,None,:].expand(-1,15,-1))
        result = scope['initialize_reference_points'](dummy, torch.zeros(1,60,256), [{}], method='sample_space')
        self.assertEqual(tuple(result.shape), (1,60,3))
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaises(ValueError):
            scope['initialize_reference_points'](dummy, torch.zeros(1,60,256), [{}], method='gt_noise')

    def test_feature_extraction_view_major_and_level_order(self):
        scope = self.methods('extract_view_features')
        dummy = types.SimpleNamespace(backbone=lambda x,level:[x,x+10], use_feat_level=[0,1])
        result = scope['extract_view_features'](dummy, [torch.zeros(1,3,2,2),torch.ones(1,3,2,2)])
        torch.testing.assert_close(result[0][:,0,0,0], torch.tensor([10.,11.]))

    def test_precomputed_payload_reaches_decoder_without_backbone(self):
        scope = self.methods('forward')
        scope['time'] = time
        for name in ('time_backbone', 'time_preprocess', 'time_init_ref'):
            scope[name] = types.SimpleNamespace(update=lambda value: None)
        received = {}
        class ReachedDecoder(Exception):
            pass
        def decoder(tgt, reference_points, features, **kwargs):
            received.update(features=features, metadata=kwargs['meta'])
            raise ReachedDecoder()
        def forbidden(views):
            self.fail('Backbone ran again despite precomputed features')
        dummy = types.SimpleNamespace(
            extract_view_features=forbidden, num_instance=1, num_joints=15,
            query_embed_type='per_joint', query_embed=torch.nn.Embedding(15,512),
            init_ref_method='sample_space', init_ref_method_value=0,
            close_pose_embedding=False, gt_match=False,
            get_valid_ratio=lambda mask:torch.ones(mask.shape[0],2),
            initialize_reference_points=lambda tgt, meta, **kwargs:torch.zeros(1,15,3),
            decoder=decoder)
        payload = [torch.randn(2,256,2,2)]
        metadata = [{'id':'00_00'}, {'id':'00_02'}]
        with self.assertRaises(ReachedDecoder):
            scope['forward'](dummy, views=[torch.zeros(1,3,8,8)]*2,
                             meta=metadata, precomputed_features=payload)
        self.assertIs(received['features'][0], payload[0])
        self.assertIs(received['metadata'], metadata)
        with self.assertRaises(ValueError):
            scope['forward'](dummy, views=[torch.zeros(1,3,8,8)]*2,
                             meta=metadata, precomputed_features=[torch.zeros(3,256,2,2)])


if __name__ == '__main__':
    unittest.main()
