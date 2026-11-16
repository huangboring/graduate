from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
import json
import random
import numpy as np
import torch

from .networks import SelectorNetwork
from .runtime import SelectionEngine
from .costs import SubsetCostEvaluator, mpjpe
from .selection import AcquisitionConfig
from .types import WarmupError
from .replay import ReplayProvider, load_sequence, ReplaySequences


@dataclass
class TrainConfig:
    memory_type: str = 'gru'
    memory_window: int = 32
    epochs: int = 8
    hidden: int = 64
    lr: float = 0.001
    tbptt: int = 16
    seed: int = 7
    history_limit: int = 256
    device: str = 'cpu'
    training_only: bool = False
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)


def sequence_paths(path):
    if isinstance(path,(list,tuple)):
        paths=[Path(p) for p in path]
        if not paths or not all(p.is_file() for p in paths):
            raise ValueError('Protocol contains missing sequence files')
        return paths
    p = Path(path)
    paths = sorted(p.glob('*.npz')) if p.is_dir() else [p]
    if not paths or not all(x.is_file() for x in paths):
        raise ValueError('Expected an NPZ sequence or directory containing NPZ sequences')
    return paths


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def expected_mpjpe_loss(logits: torch.Tensor, costs_m: torch.Tensor) -> torch.Tensor:
    """Exact finite-action (subset) expectation for a FIXED acquired history.

    Costs are reconstruction MPJPE in metres, not regression targets. Detaching
    costs freezes the reconstruction route; gradients update only the selector.
    This is not a long-horizon policy gradient through earlier discrete actions.
    """
    if logits.ndim != 1 or costs_m.shape != logits.shape or logits.numel() < 1:
        raise ValueError('Expected matching nonempty 1D logits and candidate costs')
    costs = costs_m.detach().to(device=logits.device, dtype=logits.dtype)
    if not torch.isfinite(logits).all() or not torch.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Logits must be finite and MPJPE costs finite/nonnegative')
    return (torch.softmax(logits, dim=0) * costs).sum()


def backend_provenance(backend):
    return getattr(backend, 'provenance', {'backend':type(backend).__name__})


def train(data, output, backend, config: TrainConfig, validation_data=None, allow_benchmark_inputs=False):
    if config.epochs < 1 or config.tbptt < 2 or config.lr <= 0:
        raise ValueError('Positive epochs/lr and tbptt >= 2 required')
    if config.history_limit < getattr(backend, 'window_size', 1):
        raise ValueError('history_limit must cover the reconstruction window')
    paths = sequence_paths(data)
    vpaths = sequence_paths(validation_data) if validation_data else []
    from .protocol import audit_partitions, sha256_file
    partition_manifest = audit_partitions({'train': paths, 'validation': vpaths})
    if backend_provenance(backend).get('backend') == 'local_aux_v1':
        from .local_training import check_backend_splits
        check_backend_splits(backend,partition_manifest)
    real_data = False
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            real_data |= str(archive['input_provenance'].item()) != 'synthetic_demo'
    if all(len(r['source_ids']) <= config.acquisition.max_sources for r in partition_manifest['train']):
        raise ValueError('No subset choice: every training roster fits the budget. Reduce max_sources or supply more candidates; do not pretend all-view acquisition trains selection.')
    if config.training_only and vpaths:
        raise ValueError('Training-only mode cannot report validation')
    if real_data and not vpaths and not config.training_only:
        raise ValueError('Real-data training requires a disjoint validation split')
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    sequences = ReplaySequences(paths, allow_benchmark_inputs)
    validation = ReplaySequences(vpaths, allow_benchmark_inputs)
    joints = len(sequences[0][0].observation.xy)
    network = SelectorNetwork(joints, config.hidden,config.memory_type,config.memory_window).to(config.device)
    network.acquisition_config = config.acquisition
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr)
    evaluator = SubsetCostEvaluator(backend)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'selector.pt').exists():
        raise FileExistsError('Output already contains selector.pt; choose a new run directory')
    manifest = {'config':asdict(config),'backend':type(backend).__name__,
                'experiment_scope':'training_only_no_validation_or_test' if config.training_only else 'scene_disjoint',
                'backend_provenance':backend_provenance(backend),
                'allow_benchmark_inputs':allow_benchmark_inputs,
                'train_files':[{ 'path':str(p.resolve()),'sha256':sha256_file(p)} for p in paths],
                'validation_files':[str(p.resolve()) for p in vpaths],
                'training_data_manifest':partition_manifest,
                'cost':'fixed-backend MPJPE in metres; no implicit alignment',
                'selection_objective':'sum(softmax(subset_logits) * detached_joint_subset_mpjpe)',
                'subset_policy':'additive_fixed_size_conditioned_on_refresh',
                'gradient_scope':'one-step conditional on acquired history; truncated BPTT through memory, no long-horizon action gradient',
                'rollin':'categorical sample from current policy; no GT-based action selection'}
    _save_json(output/'manifest.json',manifest)
    records, best = [], float('inf')
    for epoch in range(config.epochs):
        network.train()
        running, nlabels, chosen_errors, updates, trainable_frames = 0.0, 0, [], 0, 0
        order = list(range(len(sequences))); rng.shuffle(order)
        for sequence_index in order:
            engine = SelectionEngine(network,config.history_limit,config.acquisition)
            sequence = sequences[sequence_index]
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for t, frame in enumerate(sequence):
                if t%100==0: print(f'selector epoch {epoch+1}: frame {t+1}/{len(sequence)}',flush=True)
                # Fix the WHOLE batch before looking at any current packet/cost.
                decision = engine.score(frame.observation,frame.candidates,frame.geometry,sample=True,rng=rng)
                distribution = engine.subset_distribution(decision)
                try:
                    costs = evaluator.evaluate(frame.observation, distribution.subsets, frame.packets,
                                           tuple(engine.history),frame.gt,frame.gt_valid,frame.coordinate_frame)
                except WarmupError:
                    costs = None
                if costs is not None:
                    target = torch.as_tensor(costs.errors_m,device=config.device)
                    loss = expected_mpjpe_loss(distribution.logits,target)
                    # A fully forced/all-view action has no selection gradient.
                    # Do not apply optimizer weight decay on such frames alone.
                    if len(distribution.subsets) > 1:
                        losses.append(loss); trainable_frames += 1
                    running += float(loss.detach()); nlabels += 1
                if costs is not None:
                    chosen_errors.append(float(costs.errors_m[distribution.subsets.index(decision.selected_ids)]))
                provider = ReplayProvider(frame)
                packets = provider.acquire_many(decision.selected_ids,frame.observation.target_id,frame.observation.timestamp)
                engine.commit(frame.observation,packets,backend)
                if (t+1) % config.tbptt == 0 or t+1 == len(sequence):
                    if losses:
                        torch.stack(losses).mean().backward()
                        torch.nn.utils.clip_grad_norm_(network.parameters(),1.0)
                        optimizer.step(); updates += 1
                    optimizer.zero_grad(set_to_none=True)
                    engine.detach_memory()
                    losses = []
            # New session/target starts with empty memories, never all-view history.
        if nlabels == 0:
            raise ValueError('No evaluable candidate costs: sequences may be shorter than the backend warmup window')
        if trainable_frames == 0:
            raise ValueError('No trainable subset choices after warmup: every action was fully forced. Check refresh interval, slots, budget and sequence duration.')
        row = {'epoch':epoch+1,'expected_mpjpe_m':running/nlabels,'cost_evaluated_frames':nlabels,
               'trainable_subset_frames':trainable_frames,
               'optimizer_updates':updates,'rollin_mpjpe_mm':float(np.mean(chosen_errors)*1000)}
        checkpoint = {'version':3,'objective':'expected_subset_mpjpe','score_semantics':'additive_subset_logit',
                      'joints':joints,'hidden':config.hidden,'state_dict':network.state_dict(),
                      'epoch':epoch+1,'config':asdict(config),'backend':type(backend).__name__,
                      'backend_provenance':backend_provenance(backend),
                      'training_data_manifest':partition_manifest}
        if validation:
            report = evaluate(network,validation,backend,['learned'],config.seed,config.history_limit)
            metric = report['policies']['learned']['scene_macro_mpjpe_mm']
            if metric is None:
                raise ValueError('Validation has no evaluable reconstruction frames')
            row['validation_mpjpe_mm'] = metric
            if metric < best:
                best = metric; torch.save(checkpoint,output/'best.pt')
        torch.save(checkpoint,output/'selector.pt')
        records.append(row)
        _save_json(output/'training.json',records)
        print(json.dumps(row),flush=True)
    return network,records


def load_network(path, device='cpu'):
    checkpoint = torch.load(path,map_location=device,weights_only=True)
    if checkpoint.get('version') != 3 or checkpoint.get('objective') != 'expected_subset_mpjpe':
        raise ValueError('Expected v3 expected-subset-MPJPE checkpoint. v1 error-regression and v2 single-source checkpoints are incompatible; retrain explicitly.')
    cfg = checkpoint.get('config',{})
    net = SelectorNetwork(checkpoint['joints'],checkpoint['hidden'],cfg.get('memory_type','gru'),cfg.get('memory_window',32)).to(device)
    net.load_state_dict(checkpoint['state_dict'],strict=True)
    net.training_data_manifest = checkpoint.get('training_data_manifest')
    net.backend_provenance = checkpoint.get('backend_provenance')
    if 'acquisition' not in cfg:
        raise ValueError('Checkpoint lacks acquisition/refresh configuration')
    net.acquisition_config = AcquisitionConfig(**cfg['acquisition'])
    net.eval()
    return net


@torch.no_grad()
def evaluate(network, sequences, backend, policies=None, seed=7, history_limit=256):
    expected_backend = getattr(network,'backend_provenance',None)
    if expected_backend is not None and expected_backend != backend_provenance(backend):
        raise ValueError('Evaluation backend differs from checkpoint reconstruction provenance')
    if history_limit < getattr(backend,'window_size',1):
        raise ValueError('history_limit must cover the reconstruction window')
    policies = policies or ['learned','random','round_robin','last','geometry','all']
    if not set(policies) <= {'learned','random','round_robin','last','geometry','all'}:
        raise ValueError('Unknown evaluation policy')
    acquisition = getattr(network,'acquisition_config',AcquisitionConfig())
    network.eval()
    result = {'policies':{},'metric':'MPJPE mm; matching supplied coordinate frame',
        'acquisition':asdict(acquisition),
        'vertical_axis':'z' if backend_provenance(backend).get('backend')=='egoexomocap' else 'y',
        'vertical_direction':'down' if backend_provenance(backend).get('backend')=='local_aux_v1' else 'up',
        'oracle':'Minimum legal subset including mandatory refresh under EACH policy history; not a global or unconstrained oracle',
        'subset_policy':'additive_fixed_size_conditioned_on_refresh',
        'simulation_notice':'synthetic_demo results validate plumbing only'}
    for policy in policies:
        rng = random.Random(seed)
        errors, regrets, oracle_errors, expectations, records = [],[],[],[],[]
        sequence_scenes, coverage = {},{}
        requests = batches = warmup = refresh_requests = refresh_frames = 0
        max_wait = max_overdue = 0.0
        for seq_index, sequence in enumerate(sequences):
            sequence_scenes[seq_index] = sequence[0].metadata.get('scene_id',f'sequence_{seq_index}')
            largest_roster = max(len(f.candidates) for f in sequence)
            config = (replace(acquisition,max_sources=largest_roster,
                              refresh_slots=min(acquisition.refresh_slots,largest_roster))
                      if policy=='all' else acquisition)
            engine = SelectionEngine(network,history_limit,config)
            evaluator = SubsetCostEvaluator(backend)
            for frame in sequence:
                decision = engine.score(frame.observation,frame.candidates,frame.geometry,
                    policy='learned' if policy=='all' else policy,rng=rng)
                distribution = engine.subset_distribution(decision)
                try:
                    costs = evaluator.evaluate(frame.observation,distribution.subsets,frame.packets,
                        tuple(engine.history),frame.gt,frame.gt_valid,frame.coordinate_frame)
                except WarmupError:
                    costs = None
                provider = ReplayProvider(frame)
                packets = provider.acquire_many(decision.selected_ids,frame.observation.target_id,frame.observation.timestamp)
                prediction = engine.commit(frame.observation,packets,backend)
                requests += len(provider.requests); batches += len(provider.batch_requests)
                refresh_requests += len(decision.refresh.required_ids)
                refresh_frames += int(bool(decision.refresh.required_ids))
                max_wait = max(max_wait,max(decision.refresh.ages_s))
                max_overdue = max(max_overdue,max(decision.refresh.overdue_s))
                row = {'sequence':seq_index,'timestamp':frame.observation.timestamp,
                    'selected_ids':list(decision.selected_ids),'budget':decision.budget,
                    'required_refresh_ids':list(decision.refresh.required_ids),
                    'refresh_reasons':list(decision.refresh.reasons),
                    'source_wait_s':list(decision.refresh.ages_s),'source_seen':list(decision.refresh.seen),
                    'source_overdue_s':list(decision.refresh.overdue_s),
                    'demand_vector':decision.demand_vector.cpu().tolist(),
                    'candidate_ids':list(frame.candidates),'source_logits':decision.logits.cpu().tolist(),
                    'legal_subsets':[list(s) for s in distribution.subsets],
                    'subset_probabilities':distribution.probabilities.cpu().tolist()}
                if prediction is None:
                    warmup += 1; row['status']='warmup'
                else:
                    if costs is None:
                        raise RuntimeError('Subset evaluation and committed reconstruction disagree on readiness')
                    error = mpjpe(prediction,frame.gt,frame.gt_valid,frame.coordinate_frame)
                    selected_cost = float(costs.errors_m[distribution.subsets.index(decision.selected_ids)])
                    if not np.isclose(error,selected_cost,rtol=1e-5,atol=1e-7):
                        raise RuntimeError('Offline subset cost differs from actual committed reconstruction')
                    oracle = float(costs.errors_m.min())
                    expected_error = float(np.dot(distribution.probabilities.cpu().numpy(),costs.errors_m))
                    errors.append(error); regrets.append(error-oracle); oracle_errors.append(oracle); expectations.append(expected_error)
                    row.update(status='ok',mpjpe_mm=error*1000,oracle_mpjpe_mm=oracle*1000,
                        selector_expected_mpjpe_mm=expected_error*1000,subset_costs_m=costs.errors_m.tolist(),
                        joints=prediction.joints.tolist(),gt=frame.gt.tolist())
                records.append(row)
            coverage[str(seq_index)] = {'known_sources':len(engine.first_seen),
                'acquired_sources':len(engine.memories),
                'never_acquired':sorted(set(engine.first_seen)-set(engine.memories)),
                'acquisition_counts':{s:engine.memories[s].count if s in engine.memories else 0 for s in engine.first_seen}}
        summary = {'mpjpe_mm':float(np.mean(errors)*1000) if errors else None,
            'selector_expected_mpjpe_mm':float(np.mean(expectations)*1000) if expectations else None,
            'one_step_oracle_mm':float(np.mean(oracle_errors)*1000) if oracle_errors else None,
            'regret_mm':float(np.mean(regrets)*1000) if regrets else None,
            'requests':requests,'batch_requests':batches,'refresh_requests':refresh_requests,
            'refresh_frames':refresh_frames,'max_source_wait_s':max_wait,'max_overdue_s':max_overdue,
            'source_coverage':coverage,'warmup_frames':warmup,'evaluated_frames':len(errors),
            'budget_note':'all available views (different budget)' if policy=='all' else 'same configured budget and refresh rule',
            'frames':records}
        by_scene = {}
        for i,scene in sequence_scenes.items():
            values = [r['mpjpe_mm'] for r in records if r['sequence']==i and r['status']=='ok']
            if values:
                by_scene.setdefault(scene,[]).append(float(np.mean(values)))
        scene_means = {s:float(np.mean(v)) for s,v in by_scene.items()}
        summary['per_scene_mpjpe_mm'] = scene_means
        summary['scene_macro_mpjpe_mm'] = float(np.mean(list(scene_means.values()))) if scene_means else None
        result['policies'][policy] = summary
    return result
