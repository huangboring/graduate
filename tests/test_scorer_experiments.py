import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from observer_selection.fit_triangulated import AppearanceSelector, choose
from observer_selection.replay import ReplayFrame
from observer_selection.scorer_experiments import restrict_sources, load_transfer, evaluate_policies


class ScorerExperimentTest(unittest.TestCase):
    def test_mlp_batch_matches_previous_scores(self):
        net=AppearanceSelector('none')
        q=torch.randn(64)
        h=[torch.randn(64) for _ in range(4)]
        g=[torch.randn(9) for _ in h]
        ages=[0.,.1,.3,2.]
        expected=torch.stack([net.score(q,x,y,t,True) for x,y,t in zip(h,g,ages)])
        torch.testing.assert_close(net.score_candidates(q,h,g,ages),expected)

    def test_attention_camera_order_equivariance_and_gradients(self):
        torch.manual_seed(5)
        net=AppearanceSelector('none','attention')
        q=torch.randn(64,requires_grad=True)
        h=[torch.randn(64,requires_grad=True) for _ in range(4)]
        g=[np.arange(9,dtype='f4')+i for i in range(4)]
        ages=[0.,.1,.3,2.]
        scores,local=net.score_candidates(q,h,g,ages)
        order=[2,0,3,1]
        permuted,new_local=net.score_candidates(q,[h[i] for i in order],[g[i] for i in order],[ages[i] for i in order])
        torch.testing.assert_close(permuted,scores[order],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(new_local,local)
        ((scores-local)**2).sum().backward()
        self.assertGreater(float(net.score_attention.in_proj_weight.grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(all(x.grad is not None for x in h))
        # Variable camera count needs no parameter reshaping.
        self.assertEqual(net.score_candidates(q,h[:2],g[:2],ages[:2])[0].shape,(2,))

    def test_restriction_removes_unrequested_packet_access(self):
        f=ReplayFrame(None,('cam01','cam02','cam03','cam04'),{},
                      {f'cam{i:02}':object() for i in range(1,5)},None,None,'world')
        reduced=restrict_sources([f],['cam01','cam02'])[0]
        self.assertEqual(reduced.candidates,('cam01','cam02'))
        self.assertEqual(set(reduced.packets),{'cam01','cam02'})
        self.assertIn('cam03',f.packets)
        for roster in [['cam01','cam01'],['cam09'],[]]:
            with self.assertRaises(ValueError):
                restrict_sources([f],roster)

    def test_transfer_loads_weights_without_optimizer_and_records_overlap(self):
        net=AppearanceSelector('none','attention')
        model=SimpleNamespace(device='cpu',adapter=torch.nn.Linear(52,512,bias=False))
        contract={'appearance':'none','joint_weights':'vitpose','scorer':'attention',
                  'checkpoint_sha256':'official','target_id':'aria01','requester_id':'aria02',
                  'cost_m_per_source':.002,'objective':'exact_subset_expected_cost',
                  'sequence':'same','sources':['aria03','aria04','cam01','cam02']}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'experiment.json').write_text(json.dumps(contract),encoding='utf-8')
            torch.save({'selector':net.state_dict(),'adapter':model.adapter.state_dict(),'epoch':50},root/'checkpoint.pt')
            original={k:v.clone() for k,v in net.state_dict().items()}
            out=root/'evaluation'
            load_transfer(root,net,model,contract,['aria03','aria04','cam03','cam04'],'same',out)
            for key,value in net.state_dict().items():
                torch.testing.assert_close(value,original[key])
            self.assertTrue(all(not p.requires_grad for p in net.parameters()))
            protocol=json.loads((out/'evaluation_protocol.json').read_text())
            self.assertEqual(protocol['unseen_sources'],['cam03','cam04'])
            self.assertFalse(protocol['training_performed'])
            self.assertEqual(protocol['epoch'],50)
            wrong=dict(contract,target_id='aria05')
            with self.assertRaisesRegex(ValueError,'target_id'):
                load_transfer(root,net,model,wrong,['cam03'],'same',root/'bad')

    def test_train_resume_milestones_shared_labels_and_camera_transfer(self):
        from unittest.mock import patch
        from observer_selection.fit_triangulated import train
        from observer_selection.types import Observation, Packet
        frames=[]
        point=np.array([0.,0.,4.],dtype='f4')
        k=np.array([[600,0,320],[0,600,240],[0,0,1]],dtype='f4')
        for i in range(7):
            camera=np.eye(4,dtype='f4')
            obs=Observation('T',i*.05,np.tile([320.,240.],(13,1)),np.ones(13,bool),
                np.array([0,0,640,480]),np.ones(13),backend_context={
                'local_camera_to_world':camera,'local_intrinsics':k})
            packets={}
            for n in range(1,5):
                external=camera.copy();external[0,3]=n*.3
                xy=np.tile([320-600*n*.3/4,240.],(13,1))
                source=f'cam{n:02}'
                packets[source]=Packet(source,'T',i*.05,xy,obs.valid,obs.bbox,obs.confidence,
                                       {'camera_to_world':external,'intrinsics':k})
            frames.append(ReplayFrame(obs,tuple(packets),{},packets,None,None,'world'))
        class Model:
            window,device=3,'cpu'
            def __init__(self,*args):
                self.adapter=torch.nn.Linear(39,66)
            def load_tracking(self,*args):
                return np.zeros((6,54),'f4'),np.tile(point,(7,22,1)),np.tile(np.eye(3),(7,1,1))
            def reconstruct(self,sparse,head,values,*args):
                return self.adapter(torch.tensor(values.reshape(3,39))).reshape(3,22,3)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prepared=root/'prepared';prepared.mkdir()
            sequence=root/'scene'/'sequence';sequence.mkdir(parents=True)
            (prepared/'manifest.json').write_text(json.dumps({'scene_id':'scene','sequence_id':'sequence',
                'requester_id':'local','target_id':'T'}),encoding='utf-8')
            (prepared/'replay.npz').write_bytes(b'fixture replay hash')
            for source in ['local',*frames[0].candidates]:
                pair=prepared/source;pair.mkdir()
                (pair/'meta.json').write_text(json.dumps({'raw_frame_ids':list(range(1,8))}),encoding='utf-8')
                np.savez(pair/'observer.npz',observer_img_size=np.array([640,480]))
            official=root/'official.pt';official.write_bytes(b'fixture official hash')
            common=dict(prepared=str(prepared),sequence=str(sequence),repo=str(root),checkpoint=str(official),
                        device='cpu',epochs=1,cost=.002,max_windows=None,resume=False,render=False,
                        appearance='none',joint_weights='vitpose',sources=['cam01','cam02'],save_every=1,label_stride=1)
            with patch('observer_selection.fit_triangulated.SingleExo',Model), \
                 patch('observer_selection.fit_triangulated.load_sequence',return_value=frames), \
                 patch('observer_selection.preparation_quality.check_preparation',return_value={}), \
                 patch('observer_selection.selection_video.export_outputs'):
                base=root/'base';train(output=str(base),**common)
                mlp=root/'mlp'
                train(output=str(mlp),ranking_from=str(base),subset_training=True,**common)
                attention=root/'attention'
                options=dict(common,scorer='attention',ranking_from=str(base),subset_training=True,
                             target_cache_from=str(mlp))
                train(output=str(attention),**options)
                train(output=str(attention),**dict(options,epochs=2,resume=True))
                state=torch.load(attention/'checkpoint.pt',weights_only=True)
                self.assertEqual(state['epoch'],2)
                self.assertEqual(len(state['history']),2)
                self.assertTrue((attention/'epoch_0001/metrics.json').exists())
                self.assertTrue((attention/'epoch_0002/metrics.json').exists())
                self.assertFalse((attention/'subset_cost_cache').exists())
                test=root/'transfer'
                train(output=str(test),**dict(common,scorer='attention',sources=['cam03','cam04'],
                      evaluate_from=str(attention/'epoch_0002')))
                result=json.loads((test/'camera_transfer.json').read_text())
                self.assertEqual(result['sources'],['cam03','cam04'])
                self.assertEqual(len(result['methods']),6)
                self.assertFalse((test/'checkpoint.pt').exists())
                from observer_selection.scorer_report import report
                report(root)
                self.assertGreater((root/'comparison.png').stat().st_size,1000)
                self.assertIn('attention', (root/'comparison.csv').read_text(encoding='utf-8-sig'))


if __name__=='__main__':
    unittest.main()
