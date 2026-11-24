"""Strict helpers for loading the separately pinned EgoExoMoCap repository."""
import importlib.util
import sys

import torch


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load upstream module: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def strict_weights(network, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'params' in state:
        state = state['params']
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    network.load_state_dict(state, strict=True)
    return len(state)
