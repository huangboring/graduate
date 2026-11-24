import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from observer_selection.upstream import module_from_file
from observer_selection.selection_video import export_outputs
from observer_selection.single_exo import SingleExo
from observer_selection.triangulation import triangulate_views
from observer_selection.types import Observation, Packet
from observer_selection.replay import ReplayFrame


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

    def test_original_comparison_sends_separate_tokens_and_validity_to_official_net(self):
        from dataclasses import replace

        runner=SingleExo.__new__(SingleExo)
        runner.device,runner.window='cpu',2
        point=np.array([0.,0.,4.])
        a=view(np.zeros(3),point)
        data={**a.backend_context,'local_dino_cls':np.ones(768,'f4')}
        first=replace(a,backend_context=data)
        missing=replace(first,valid=np.zeros(13,bool))
        payload={'camera_to_world':data['local_camera_to_world'],'intrinsics':data['local_intrinsics'],
                 'dino_cls':np.ones(768,'f4')}
        other=Packet('C','T',1.,a.xy,a.valid,a.bbox,a.confidence,payload)
        runner.compute_exo_rays=lambda xy,*args:np.ones((len(xy),39),'f4')
        runner.fk=None
        runner.sixd2aa=lambda x:x[:,:3]
        runner.body=lambda **kwargs:SimpleNamespace(Jtr=torch.zeros(2,22,3))

        def official(batch):
            self.assertEqual(batch['exo_rays'].shape,(1,2,2,39))
            self.assertEqual(batch['dino_cls'].shape,(1,2,2,768))
            self.assertEqual(batch['bbox_valid'][0].tolist(),[[True,True],[False,True]])
            self.assertAlmostEqual(float(batch['sparse_input'][0,0,38]),-1.)
            return {'root_orient':torch.zeros(1,2,6),'pose_body':torch.zeros(1,2,126)}

        runner.net=official
        result=runner.reconstruct_original(np.zeros((2,54),'f4'),np.zeros((2,3),'f4'),
                            np.tile(np.eye(3),(2,1,1)),[[first,missing],[other,other]],1.)
        self.assertEqual(result.shape,(2,22,3))




    def test_subset_cost_learns_to_drop_redundant_source(self):
        from observer_selection.subset_training import masks_for, expected_cost_loss
        masks=torch.as_tensor(masks_for(2))
        errors=torch.tensor([.100,.070,.072,.071])
        margins=torch.zeros(2,requires_grad=True)
        optimiser=torch.optim.Adam([margins],lr=.1)
        for _ in range(150):
            loss=expected_cost_loss(margins,errors,masks,[],.002)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        self.assertGreater(float(margins[0].detach()),0.)
        self.assertLess(float(margins[1].detach()),0.)
        mandatory=torch.zeros(2,requires_grad=True)
        expected_cost_loss(mandatory,errors,masks,[1],.002).backward()
        self.assertAlmostEqual(float(mandatory.grad[1]),0.,places=6)
        self.assertTrue(torch.isfinite(mandatory.grad).all())


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
