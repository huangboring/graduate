"""Compare conversion numerics with untouched upstream pure numpy functions."""
import ast
import unittest
from pathlib import Path
import numpy as np
from observer_selection.egohumans import build_sparse,compute_rays


class UpstreamGeometryTests(unittest.TestCase):
    def test_sparse_and_rays_match_upstream(self):
        path=Path(__file__).resolve().parents[1]/'external/EgoExoMoCap/data_loaders/dataset_nymeria.py'
        if not path.is_file():self.skipTest('Upstream source absent')
        tree=ast.parse(path.read_text(encoding='utf-8'))
        names={'_matrot2sixd_np','_build_sparse','_build_sparse_head_only','_compute_exo_rays'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        namespace={'np':np}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        rng=np.random.default_rng(10);T=6
        def rotations():
            q,_=np.linalg.qr(rng.normal(size=(T,3,3)))
            q[:,:,0]*=np.linalg.det(q)[:,None]
            return q
        tracker={}
        for part in ('head','lwrist','rwrist'):
            tracker[f'wearer_{part}_pos']=rng.normal(size=(T,3))
            tracker[f'wearer_{part}_R']=rotations()
        positions=[tracker[f'wearer_{p}_pos'].copy() for p in ('head','lwrist','rwrist')]
        for p in positions:p[:,2]-=.2
        R=[tracker[f'wearer_{p}_R'] for p in ('head','lwrist','rwrist')]
        np.testing.assert_allclose(build_sparse(tracker,.2,1),namespace['_build_sparse_head_only'](positions[0],R[0]),atol=1e-6)
        np.testing.assert_allclose(build_sparse(tracker,.2,3),namespace['_build_sparse'](positions[0],R[0],positions[1],R[1],positions[2],R[2]),atol=1e-6)
        camera={'observer_K':np.array([[300,0,200],[0,310,150],[0,0,1.]]),
                'observer_T_dev_cam':np.eye(4),'observer_head_pos':rng.normal(size=(T,3)),
                'observer_head_R':rotations()}
        camera['observer_T_dev_cam'][:3,3]=[.03,.01,-.02]
        xy=rng.uniform(0,300,size=(T,13,2));valid=np.array([1,1,0,1,0,1],bool)
        expected=namespace['_compute_exo_rays'](xy,camera['observer_K'],camera['observer_T_dev_cam'],
            camera['observer_head_pos'],camera['observer_head_R'],tracker['wearer_head_pos'],tracker['wearer_head_R'],valid)
        np.testing.assert_allclose(compute_rays(xy,valid,camera,tracker),expected,rtol=1e-5,atol=1e-6)
