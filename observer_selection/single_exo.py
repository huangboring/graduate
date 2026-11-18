"""Run the official EgoExoMoCap weights with one aggregated Exo input."""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from .official_demo import module_from_file, strict_weights
from .triangulation import _camera
from .types import Observation


class SingleExo:
    def __init__(self, repo, checkpoint, device):
        repo = Path(repo).resolve()
        config = repo/'options/test_egohumans_syn_tracker_3pt.yaml'
        body = repo/'third_party/body_models/smplh/neutral/model.npz'
        for path in (config, body, checkpoint):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        sys.path[:0] = [str(repo), str(repo/'data_prep/shared')]
        from human_body_prior.body_model.body_model import BodyModel
        from data_loaders.dataset_nymeria import _build_sparse, _compute_exo_rays
        from models.model_egoexomocap import ModelEgoExoMoCap
        from utils.utils_transform import sixd2aa

        opt = yaml.safe_load(config.read_text(encoding='utf-8'))
        opt.update(is_train=False, gpu_ids=[0] if device=='cuda' else None)
        opt['body_model']['smpl_path'] = str(repo/'third_party')
        opt['path']['models'] = str(repo/'model_zoo')
        model = ModelEgoExoMoCap(opt)
        self.net = model.get_bare_model(model.net).to(device).eval().requires_grad_(False)
        strict_weights(self.net, checkpoint)
        self.fk, self.body, self.sixd2aa = model.fk, model.bm, sixd2aa
        self.build_sparse = _build_sparse
        self.compute_exo_rays = _compute_exo_rays
        self.body_for_tracking = BodyModel(bm_fname=str(body), num_betas=10).to(device).eval()
        self.repo, self.device = repo, device
        self.window = int(opt['datasets']['test']['window_size'])
        self.body.eval().requires_grad_(False)
        if self.net.fusion.exo_token_embed.in_features != 40 or self.net.fusion.embed_dim != 512:
            raise ValueError('Expected official 39 gated ray values plus bbox validity, projected to 512')
        # Learn a residual for triangulated points. Begin at the original projection.
        self.adapter = torch.nn.Linear(52, 512, bias=False).to(device)
        torch.nn.init.zeros_(self.adapter.weight)

    @torch.no_grad()
    def load_tracking(self, sequence, target, ids):
        stage = module_from_file('_triangulation_tracking',
            self.repo/'data_prep/egohumans/extract_egohumans_pairs_stage1.py')
        rows = [np.load(Path(sequence)/'processed_data/smpl'/f'{i:05}.npy',
                        allow_pickle=True).item()[target] for i in ids]
        smpl = {key: np.stack([np.asarray(r[key], dtype='f4') for r in rows])
                for key in ('global_orient','body_pose','transl','betas')}
        if not all(np.isfinite(value).all() for value in smpl.values()):
            raise ValueError('Target tracker has nonfinite values')
        smpl['betas'] = smpl['betas'].mean(axis=0)
        joints, rotations = stage.fk_aria(smpl, self.body_for_tracking, torch.device(self.device), batch=32)
        head, left, right = stage.J_HEAD, stage.J_LWRIST, stage.J_RWRIST
        sparse = self.build_sparse(joints[:, head], rotations[:, head], joints[:, left],
                                   rotations[:, left], joints[:, right], rotations[:, right])
        # The first raw frame initializes, and the official input starts at raw frame 2.
        return sparse, joints, rotations[:, head]

    @torch.no_grad()
    def confidence(self, view):
        context = view.backend_context if hasattr(view, 'backend_context') else view.payload
        key = 'local_dino_cls' if hasattr(view, 'backend_context') else 'dino_cls'
        if key not in context:
            return view.confidence
        dino = torch.as_tensor(context[key], device=self.device).reshape(1, 1, 768)
        score = self.net.fusion.scorenet(dino).reshape(13).cpu().numpy()
        return np.where(np.asarray(view.confidence) >= .15, score, 0.)

    def reconstruct(self, sparse, head, points, state, valid, floor):
        f = self.net.fusion
        x = torch.as_tensor(sparse, device=self.device).unsqueeze(0).clone()
        x[..., [38,41,44]] -= float(floor)
        points = torch.as_tensor(points, device=self.device).reshape(1, self.window, 39)
        tri = torch.as_tensor(state==2, device=self.device).reshape(1, self.window, 13)
        valid = torch.as_tensor(valid, device=self.device).reshape(1, self.window, 1)
        head = torch.as_tensor(head, device=self.device)
        with torch.no_grad():
            ego = self.net.egonet({'sparse_input':x})
            ego_token = f.ego_token_embed(f._ego_features(x))
            if f.use_egonet:
                pos = self.fk(body_model=self.body, global_orientation=ego['root_orient'],
                              joint_rotation=ego['pose_body'], betas=None)
                ego_token = ego_token + f.egonet_token_embed(pos.reshape(1,self.window,66))
        exo_token = f.exo_token_embed(torch.cat((points, valid.to(points.dtype)), dim=-1))
        tri_points = points.reshape(1,self.window,13,3) * tri.unsqueeze(-1)
        exo_token = exo_token + self.adapter(torch.cat((tri_points.flatten(-2), tri.to(points.dtype)), dim=-1))
        tokens = torch.stack((ego_token, exo_token), dim=2).reshape(self.window,2,512)
        spatial = f.spatial_encoder(tokens)[:,0].unsqueeze(0)
        temporal = f.temporal_encoder(spatial + f.temporal_pe[:self.window].unsqueeze(0))
        root = self.sixd2aa(f.global_orientation_decoder(temporal).reshape(-1,6))
        body = self.sixd2aa(f.joint_rotation_decoder(temporal).reshape(-1,6)).reshape(self.window,63)
        joints = self.body(root_orient=root, pose_body=body).Jtr[:,:22]
        return joints - joints[:,15:16] + head[:,None]


    @torch.no_grad()
    def reconstruct_original(self, sparse, head, rotations, views, floor):
        rays, descriptors, flags = [], [], []
        for stream in views:
            cameras, intrinsics = zip(*map(_camera,stream))
            cameras = np.stack(cameras)
            if not np.allclose(intrinsics,intrinsics[0]):
                raise ValueError('The official Exo ray loader requires constant rectified intrinsics per view')
            valid = np.array([v.valid.any() for v in stream])
            rays.append(self.compute_exo_rays(np.stack([v.xy for v in stream]),intrinsics[0],
                np.eye(4),cameras[:,:3,3],cameras[:,:3,:3],head,rotations,valid))
            descriptors.append(np.stack([
                v.backend_context['local_dino_cls'] if isinstance(v,Observation) else v.payload['dino_cls']
                for v in stream]))
            flags.append(valid)
        x = torch.as_tensor(sparse,device=self.device).unsqueeze(0).clone()
        x[..., [38,41,44]] -= float(floor)
        data = {'sparse_input':x, 'fk':self.fk, 'body_model':self.body, 'betas':None}
        for name,value in (('exo_rays',rays),('dino_cls',descriptors),('bbox_valid',flags)):
            data[name] = torch.as_tensor(np.stack(value,axis=1),device=self.device).unsqueeze(0)
        out = self.net(data)
        root = self.sixd2aa(out['root_orient'].reshape(-1,6))
        body = self.sixd2aa(out['pose_body'].reshape(-1,6)).reshape(self.window,63)
        joints = self.body(root_orient=root,pose_body=body).Jtr[:,:22]
        return joints-joints[:,15:16]+torch.as_tensor(head,device=self.device)[:,None]
