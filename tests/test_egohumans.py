import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from observer_selection.egohumans import (
    export_prepared_egohumans,prepare_local_features,build_sparse,compute_rays,_hash)


class FakeVit:
    def batch(self,images,boxes):
        return [SimpleNamespace(xy=np.full((1,13,2),4,dtype='f4'),scores=np.full((1,13),.8,dtype='f4'))]


class FakeDino:
    def batch_cls(self,images):return np.ones((1,768),'f4')


class EgoHumansExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.T=4;self.ts=np.arange(self.T,dtype=np.int64)*50_000_000
        head=np.tile([0,0,1.6],(self.T,1));head[:,0]=np.arange(self.T)*.01
        self.tracker={'timestamps_ns':self.ts,'world_frame':np.array('aria01_local'),
            'wearer_head_pos':head,'wearer_head_R':np.tile(np.eye(3),(self.T,1,1))}
        self.trackpath=self.root/'tracker.npz';np.savez(self.trackpath,**self.tracker)
        self.pairs={}
        for source in ['aria02','aria03','aria04']:
            pair=self.root/source;pair.mkdir();self.pairs[source]=str(pair)
            meta={'source_dataset':'egohumans','source_scene':'04_basketball','source_subseq':'seq01',
                  'wearer_aria':'aria01','observer_aria':source,'world_frame':'aria01_local',
                  'n_frames':self.T,'t_start_ns':0,'dt_ns':50_000_000}
            (pair/'meta.json').write_text(json.dumps(meta))
            (pair/'timestamps.csv').write_text('frame_idx,t_ns\n'+''.join(f'{i},{t}\n' for i,t in enumerate(self.ts)))
            np.savez(pair/'observer.npz',observer_K=np.array([[10,0,4],[0,10,4],[0,0,1]]),
                observer_T_dev_cam=np.eye(4),observer_head_pos=np.tile([0,0,-1],(self.T,1)),
                observer_head_R=np.tile(np.eye(3),(self.T,1,1)),observer_img_size=np.array([8,8]))
            gt=np.zeros((self.T,22,3),'f4');gt[:,:,2]=np.linspace(0,1,22)
            np.savez(pair/'wearer_smplh_fk.npz',joints_pos=gt)
            boxpath=pair/'detector_boxes.npz'
            np.savez(boxpath,timestamps_ns=self.ts,last_input_t_ns=self.ts,bbox=np.tile([1,1,6,6],(self.T,1)),bbox_valid=np.array([1,1,0,1],bool))
            br=pair/'detector_receipt.json'
            br.write_text(json.dumps({'crop_source':'local_image_detector','reference_pose_used':False,
                'output_sha256':_hash(boxpath),'detector_checkpoint':'TEST_FIXTURE','identity_source':'local_tracking'}))
            prepare_local_features(pair,boxpath,br,FakeVit(),FakeDino(),
                {'vitpose':'TEST_FIXTURE','dinov3':'TEST_FIXTURE'},[np.zeros((8,8,3),np.uint8)]*self.T)
        self.manifest={'schema_version':1,'source_dataset':'egohumans','scene_id':'04_basketball',
            'sequence_id':'seq01','target_id':'aria01','requester_id':'aria02','source_ids':['aria03','aria04'],
            'pairs':self.pairs,'target_tracker':{'path':str(self.trackpath),'provenance':'measured','sha256':_hash(self.trackpath)},
            'floor':{'height_m':.1,'provenance':'calibrated'},
            'world_to_zup':{'rotation':np.eye(3).tolist(),'provenance':'calibrated'}}
        self.mp=self.root/'manifest.json';self.save_manifest()

    def save_manifest(self):self.mp.write_text(json.dumps(self.manifest))
    def tearDown(self):self.temp.cleanup()
    def export(self,name='out.npz',**kw):return export_prepared_egohumans(self.mp,self.root/name,**kw)

    def test_actual_conversion_shapes_alignment_and_invalid_frames(self):
        r=self.export();self.assertEqual(r['frames'],3)
        with np.load(r['output'],allow_pickle=False) as z:
            self.assertEqual(z['packet_exo_rays'].shape,(3,2,39))
            self.assertEqual(z['sparse_input'].shape,(3,18))
            np.testing.assert_allclose(z['sparse_input'][:,14],1.5)
            np.testing.assert_array_equal(z['timestamps'],self.ts[1:]/1e9)
            self.assertFalse(z['local_valid'][1].any())
            np.testing.assert_array_equal(z['local_bbox'][1],[0,0,8,8])
            self.assertFalse(z['packet_valid'][1].any())
            self.assertEqual(z['source_ids'].tolist(),['aria03','aria04'])
            np.testing.assert_array_equal(z['gt_joints'][:,0],0)
        self.assertEqual(json.loads(Path(r['provenance']).read_text())['dropped_initial_frames'],1)

    def test_gt_change_does_not_change_model_inputs(self):
        first=self.export('first.npz')
        for path in self.pairs.values():
            p=Path(path)/'wearer_smplh_fk.npz'
            with np.load(p) as z: gt=z['joints_pos'].copy()
            gt[:,5,0]+=100
            np.savez(p,joints_pos=gt)
        second=self.export('second.npz')
        with np.load(first['output']) as a,np.load(second['output']) as b:
            for key in ('local_xy','local_bbox','packet_xy','sparse_input','packet_exo_rays','local_exo_rays'):
                np.testing.assert_array_equal(a[key],b[key])
            self.assertFalse(np.array_equal(a['gt_joints'],b['gt_joints']))

    def test_exact_timestamp_alignment_no_silent_truncation(self):
        p=Path(self.pairs['aria03'])/'timestamps.csv'
        p.write_text(p.read_text().replace('1,50000000','1,50000001'))
        with self.assertRaisesRegex(ValueError,'timestamp'):self.export()

    def test_roster_excludes_requester_and_requires_choice(self):
        self.manifest['source_ids']=['aria02','aria03'];self.save_manifest()
        with self.assertRaisesRegex(ValueError,'external candidates'):self.export()

    def test_gt_tracking_requires_explicit_opt_in(self):
        self.manifest['target_tracker']['provenance']='benchmark_gt';self.save_manifest()
        with self.assertRaisesRegex(ValueError,'opt-in'):self.export()
        r=self.export(allow_benchmark_inputs=True)
        with np.load(r['output']) as z:self.assertEqual(z['input_provenance'].item(),'benchmark_gt_derived')

    def test_future_preprocessing_rejected(self):
        p=Path(self.pairs['aria03'])/'selector_observations.npz'
        with np.load(p) as z:a={k:z[k] for k in z.files}
        a['last_input_t_ns'][1]+=1;np.savez(p,**a)
        rp=p.with_name('selector_preparation.json');r=json.loads(rp.read_text());r['output_sha256']=_hash(p);rp.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'Future'):self.export()

    def test_gt_crops_and_unverified_egonet_rejected(self):
        p=Path(self.pairs['aria03'])/'selector_preparation.json'
        r=json.loads(p.read_text());r['crop_source']='egonet';p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'image-detector'):self.export()

    def test_input_digest_change_rejected(self):
        p=Path(self.pairs['aria03'])/'selector_observations.npz'
        p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'digest'):self.export()

    def test_no_overwrite(self):
        self.export()
        with self.assertRaises(FileExistsError):self.export()

    def test_feature_extraction_frame_count_mismatch_fails(self):
        pair=Path(self.pairs['aria03']);out=pair/'selector_observations.npz';receipt=pair/'selector_preparation.json'
        out.unlink();receipt.unlink()
        with self.assertRaisesRegex(ValueError,'frame count'):
            prepare_local_features(pair,pair/'detector_boxes.npz',pair/'detector_receipt.json',FakeVit(),FakeDino(),
                {'vitpose':'TEST_FIXTURE','dinov3':'TEST_FIXTURE'},[np.zeros((8,8,3),np.uint8)]*3)
        self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
