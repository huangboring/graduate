"""Local checkpoint adapters. Model identities include weight/config digests."""
from pathlib import Path
import json
import torch
from .egohumans import _hash, prepare_local_features


def checkpoint_identity(directory):
    root = Path(directory).resolve()
    files = [root / name for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')]
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    return json.dumps({p.name: _hash(p) for p in files}, sort_keys=True)


class DinoV2Encoder:
    def __init__(self, directory, device='cuda'):
        from transformers import AutoImageProcessor, AutoModel
        self.identity = checkpoint_identity(directory)
        self.processor = AutoImageProcessor.from_pretrained(str(directory), local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(str(directory), local_files_only=True).eval().to(device)
        if self.model.config.model_type != 'dinov2' or self.model.config.hidden_size != 768:
            raise ValueError('Expected DINOv2 base with 768-dimensional CLS features')
        self.device = device
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def batch_cls(self, images):
        batch = self.processor(images=images, return_tensors='pt').to(self.device)
        features = self.model(**batch).last_hidden_state[:, 0]
        if features.shape != (len(images), 768) or not torch.isfinite(features).all():
            raise ValueError('Invalid DINOv2 features')
        return features


def prepare_dinov2_features(pair, boxes, receipt, vitpose_dir, dinov2_dir, repo, device='cuda'):
    import importlib.util
    helper = Path(repo).resolve() / 'data_prep/shared/vitpose_predictor.py'
    spec = importlib.util.spec_from_file_location('_selector_vitpose_helper', helper)
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    vit_identity = checkpoint_identity(vitpose_dir)
    vit = module.ViTPosePredictor(model_id=Path(vitpose_dir).resolve(), device=device)
    # Pin processor behaviour rather than relying on changing HF defaults.
    from transformers import AutoImageProcessor
    vit.processor = AutoImageProcessor.from_pretrained(str(vitpose_dir), local_files_only=True, use_fast=False)
    dino = DinoV2Encoder(dinov2_dir, device)
    prepare_local_features(pair, boxes, receipt, vit, dino,
        {'vitpose': vit_identity + ';helper_sha256=' + _hash(helper),
         'dinov2': dino.identity + ';use_fast=False;adapter_sha256=' + _hash(Path(__file__))})
