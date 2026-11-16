"""Trace one simultaneous subset decision and its exact expected-MPJPE gradient.

Synthetic fixture only. Example:
python trace_one_step.py --run runs/batch_refresh_gru_v3 --output runs/batch_trace_v3.json
"""
import argparse
import json
from pathlib import Path
import torch
from observer_selection.training import load_network,expected_mpjpe_loss
from observer_selection.replay import load_sequence,ReplayProvider
from observer_selection.runtime import SelectionEngine
from observer_selection.backends import WeightedJointBackend
from observer_selection.costs import SubsetCostEvaluator
from observer_selection.features import demand_features
from observer_selection.protocol import verify_heldout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);run=Path(args.run)
    if out.exists():raise FileExistsError('Choose a new trace path')
    torch.set_num_threads(4)
    net=load_network(run/'best.pt');backend=WeightedJointBackend()
    verify_heldout(net,[run/'test.npz'],backend)
    frames=load_sequence(run/'test.npz')
    if len(frames)<3 or frames[0].metadata['input_provenance']!='synthetic_demo':
        raise ValueError('This trace requires at least three synthetic frames')
    engine=SelectionEngine(net);actions=[]
    for f in frames[:2]:
        d=engine.score(f.observation,f.candidates,f.geometry,policy='round_robin')
        actions.append(list(d.selected_ids))
        packets=ReplayProvider(f).acquire_many(d.selected_ids,f.observation.target_id,f.observation.timestamp)
        engine.commit(f.observation,packets,backend)
    f=frames[2];d=engine.score(f.observation,f.candidates,f.geometry)
    distribution=engine.subset_distribution(d)
    d.logits.retain_grad();distribution.logits.retain_grad()
    costs=SubsetCostEvaluator(backend).evaluate(f.observation,distribution.subsets,f.packets,
        tuple(engine.history),f.gt,f.gt_valid,f.coordinate_frame)
    loss=expected_mpjpe_loss(distribution.logits,torch.tensor(costs.errors_m))
    analytic=distribution.probabilities.detach()*(torch.tensor(costs.errors_m)-loss.detach())
    loss.backward();torch.testing.assert_close(distribution.logits.grad,analytic)
    report={'notice':'SYNTHETIC ONLY: WeightedJointBackend, NOT EgoExoMoCap accuracy.',
        'objective':'expected_joint_subset_mpjpe','memory_type':net.memory_type,
        'prior_batches':actions,'timestamp':f.observation.timestamp,
        'selected_ids':list(d.selected_ids),'required_refresh_ids':list(d.refresh.required_ids),
        'refresh_reasons':list(d.refresh.reasons),'source_logits':d.logits.detach().tolist(),
        'source_logit_gradients':d.logits.grad.tolist(),
        'demand_feature_dim':len(demand_features(f.observation,frames[1].observation)),
        'q_first8':d.demand_vector.detach().tolist()[:8],
        'subsets':[{'ids':list(s),'probability':float(distribution.probabilities[i].detach()),
            'joint_mpjpe_m':float(costs.errors_m[i]),'subset_logit_gradient':float(distribution.logits.grad[i])}
            for i,s in enumerate(distribution.subsets)],
        'loss_m':float(loss.detach()),'analytic_gradient_matches':True,
        'gradient_norms':{n:float(p.grad.norm()) for n,p in net.named_parameters() if p.grad is not None}}
    provider=ReplayProvider(f)
    packets=provider.acquire_many(d.selected_ids,f.observation.target_id,f.observation.timestamp)
    engine.commit(f.observation,packets,backend)
    report['batch_requests']=[list(ids) for ids in provider.batch_requests]
    report['source_requests']=provider.requests
    report['history_after']={s:{'count':m.count,'timestamp':m.last_timestamp} for s,m in engine.memories.items()}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'output':str(out.resolve()),'selected_ids':d.selected_ids,
        'analytic_gradient_matches':True,'synthetic_only':True}))


if __name__=='__main__':main()
