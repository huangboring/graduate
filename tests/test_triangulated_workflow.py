import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from observer_selection.fit_triangulated import AppearanceSelector, _run, choose, matched_random
from observer_selection.official_demo import module_from_file
from observer_selection.selection_video import export_outputs
from observer_selection.single_exo import SingleExo
from observer_selection.triangulation import triangulate_views
from observer_selection.types import Observation, Packet
from observer_selection.replay import ReplayFrame
from observer_selection.selection import AcquisitionConfig


def view(center, point, mask=None):
    camera = np.eye(4)
    camera[:3,3] = center
    k = np.array([[600,0,320],[0,600,240],[0,0,1]],dtype='f4')
    x = k @ (point-center)
    xy = np.tile(x[:2]/x[2], (13,1))
    return Observation('T',1.,xy,np.ones(13,bool) if mask is None else mask,
                       np.array([1,1,640,480]),np.ones(13),
                       backend_context={'local_camera_to_world':camera,'local_intrinsics':k})


class TriangulatedPipelineTest(unittest.TestCase):
    def test_all_views_for_valid_joints_and_one_view_fallback(self):
        point = np.array([.3,.25,4.])
        head = np.array([0.,0.,4.])
        mask = np.ones(13,bool)
        mask[1] = False
        views = [view(np.array([0.,0.,0.]),point),
                 view(np.array([1.,0.,0.]),point,mask),
                 view(np.array([-1.,0.,0.]),point,mask)]
        points, states = triangulate_views(views,head,np.eye(3),np.ones((3,13)))
        self.assertEqual(list(states[:3]),[2,1,2])
        np.testing.assert_allclose(points[0],point-head,atol=1e-6)
        self.assertFalse(np.allclose(points[1],points[0]))

    def test_local_threshold_and_mandatory_refresh(self):
        cameras=('C','D','E','F','G')
        scores=torch.tensor([.9,.8,.7,.6,.4],requires_grad=True)
        local=torch.tensor(.6,requires_grad=True)
        self.assertEqual(choose(scores,local,cameras,('G',),'learned',None)[0],('C','D','E','G'))
        self.assertEqual(choose(scores,torch.tensor(1.),cameras,(),'learned',None)[0],())
        _, logp=choose(scores,local,cameras,('G',),'sample',random.Random(1))
        logp.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(float(scores.grad[-1]),0.)
        self.assertNotEqual(float(local.grad),0.)
        selected=matched_random(cameras,{'selected':['C','D','G'],'refresh':['G']},random.Random(5))
        self.assertIn('G',selected)
        self.assertEqual(len(selected),3)

    def test_official_fusion_interface_gradient(self):
        repo=Path(__file__).resolve().parents[1]/'external/EgoExoMoCap'
        source=repo/'networks/egoexomocap.py'
        if not source.is_file():
            self.skipTest('Official repository is absent')
        upstream=module_from_file('_test_single_exo_upstream',source)
        fusion=upstream.FusionNet(input_dim_ego=54,num_input=3,window_size=3,embed_dim=512,nhead=8,
                                  num_layer_spatial=1,num_layer_temporal=1,egonet_out_dim=66)
        fusion.requires_grad_(False).eval()
        runner=SingleExo.__new__(SingleExo)
        runner.net=SimpleNamespace(fusion=fusion,egonet=lambda batch:{
            'root_orient':torch.zeros(1,3,6),'pose_body':torch.zeros(1,3,126)})
        runner.fk=lambda **kw:torch.zeros(1,3,22,3)
        runner.sixd2aa=lambda x:x[:,:3]
        runner.body=lambda root_orient,pose_body:SimpleNamespace(
            Jtr=torch.cat((root_orient[:,None],pose_body.reshape(3,21,3)),dim=1))
        runner.device,runner.window='cpu',3
        runner.adapter=torch.nn.Linear(52,512,bias=False)
        result=runner.reconstruct(np.zeros((3,54),'f4'),np.zeros((3,3),'f4'),
                                  np.ones((3,13,3),'f4'),np.full((3,13),2),np.ones(3,bool),0.)
        self.assertEqual(tuple(result.shape),(3,22,3))
        result.square().mean().backward()
        self.assertGreater(float(runner.adapter.weight.grad.abs().sum()),0.)

    def test_two_training_windows_with_memory_and_one_token(self):
        from dataclasses import replace
        points=np.array([0.,0.,4.])
        frames=[]
        for i in range(7):
            obs=replace(view(np.zeros(3),points),timestamp=i*.05)
            other=view(np.array([1.,0.,0.]),points)
            packet=Packet('C','T',obs.timestamp,other.xy,other.valid,other.bbox,
                          other.confidence,{'camera_to_world':other.backend_context['local_camera_to_world'],
                                            'intrinsics':other.backend_context['local_intrinsics']})
            frames.append(ReplayFrame(obs,('C',),{}, {'C':packet},np.zeros((22,3)),
                                      np.ones(22,bool),'world'))
        class Model:
            window, device = 3, 'cpu'
            adapter = torch.nn.Linear(39,66)
            @staticmethod
            def confidence(obs):
                return obs.confidence
            def reconstruct(self,sparse,head,values,state,valid,floor):
                return self.adapter(torch.from_numpy(values.reshape(3,39))).reshape(3,22,3)
        model=Model()
        selector=AppearanceSelector()
        before=selector.scorer[-1].weight.detach().clone()
        data=(frames,list(range(1,8)),{'A':640,'C':640},set(),'A',np.zeros((6,54),'f4'),
              np.tile(points.astype('f4'),(7,22,1)),np.tile(np.eye(3),(7,1,1)))
        optimiser=torch.optim.Adam([*selector.parameters(),*model.adapter.parameters()])
        result,predictions,_=_run(data,model,selector,AcquisitionConfig(max_sources=1),
                                   'sample',11,2,optimiser)
        self.assertEqual([row['frame'] for row in result],list(range(2,8)))
        self.assertEqual(predictions.shape,(6,22,3))
        self.assertFalse(torch.equal(before,selector.scorer[-1].weight))

    def test_multiview_animation_and_figures(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            image=out/'frame.jpg'
            cv2.imwrite(str(image),np.full((100,150,3),220,'u1'))
            images={s:{1:image,2:image,3:image} for s in ('A','C','D')}
            rows=[{'frame':2,'mpjpe_mm':23.,'selected':['D'],'triangulated_joints':5},
                  {'frame':3,'mpjpe_mm':20.,'selected':[],'triangulated_joints':0}]
            warmup={'frame':1,'selected':['C','D'],'mpjpe_mm':None,'triangulated_joints':None}
            export_outputs(out,[{'epoch':1,'mpjpe_mm':25.}],rows,rows,rows,warmup,('C','D'),'A',
                           np.zeros((2,22,3),'f4'),images,0,
                           joints_for=lambda source,frame: [(75,50)] if source=='D' else [])
            self.assertTrue((out/'figures/overview.png').is_file())
            reader=cv2.VideoCapture(str(out/'multiview_selection.mp4'))
            try:
                self.assertTrue(reader.read()[0])
                ok,frame=reader.read()
                self.assertTrue(ok)
                self.assertGreater(float(frame[440,930].mean()),float(frame[440,290].mean())+75.)
                green=frame[465:490,945:980].astype(int)
                self.assertTrue(np.any(green[...,1]-green[...,2]>35))
            finally:
                reader.release()


if __name__=='__main__':
    unittest.main()
