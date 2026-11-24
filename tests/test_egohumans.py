import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from observer_selection.egohumans import (
    prepare_local_features,_hash)


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










    def test_feature_extraction_frame_count_mismatch_fails(self):
        pair=Path(self.pairs['aria03']);out=pair/'selector_observations.npz';receipt=pair/'selector_preparation.json'
        out.unlink();receipt.unlink()
        with self.assertRaisesRegex(ValueError,'frame count'):
            prepare_local_features(pair,pair/'detector_boxes.npz',pair/'detector_receipt.json',FakeVit(),FakeDino(),
                {'vitpose':'TEST_FIXTURE','dinov3':'TEST_FIXTURE'},[np.zeros((8,8,3),np.uint8)]*3)
        self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
