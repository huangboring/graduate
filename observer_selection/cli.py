from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import torch

from .backends import EgoExoMoCapBackend, WeightedJointBackend
from .replay import load_sequence, make_synthetic, ReplaySequences
from .training import TrainConfig, train, evaluate, load_network, sequence_paths, _save_json
from .protocol import load_protocol, audit_partitions, verify_heldout
from .selection import AcquisitionConfig


def acquisition_args(parser):
    parser.add_argument('--max-sources',type=int,default=2,help='Maximum remote sources per frame; local view excluded')
    parser.add_argument('--refresh-after-s',type=float,default=1.0,help='Refresh threshold in seconds; engineering default, tune on validation')
    parser.add_argument('--refresh-slots',type=int,default=1,help='Slots reserved for oldest due sources when needed')
    parser.add_argument('--no-refresh',action='store_true',help='Explicit ablation; permits indefinite source starvation')
    parser.add_argument('--max-exact-subsets',type=int,default=4096,help='Fail above this exact-training/evaluation subset count')


def acquisition_from_args(args):
    return AcquisitionConfig(args.max_sources,None if args.no_refresh else args.refresh_after_s,
                             args.refresh_slots,args.max_exact_subsets)


def backend_from_args(args):
    if args.backend == 'weighted':
        return WeightedJointBackend()
    if args.backend == 'local-aux':
        from .local_fusion import LocalAuxBackend
        if not args.checkpoint or not args.body_model:
            raise ValueError('--checkpoint (trained local-aux) and --body-model are required')
        return LocalAuxBackend.from_checkpoint(args.repo,args.checkpoint,args.body_model,args.device)
    if not args.config or not args.checkpoint:
        raise ValueError('--config and --checkpoint are required for egoexo')
    return EgoExoMoCapBackend.from_repository(args.repo,args.config,args.checkpoint,args.device,
        body_model_path=args.body_model,allow_benchmark_provenance=args.allow_benchmark_inputs)


def write_viewer(report, output):
    """Standalone offline pose-point viewer. Never fetches external resources."""
    encoded = json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    html = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Batch Selection — Expected Subset MPJPE</title>
<style>body{font:16px system-ui;background:#f4f1eb;color:#24211e;max-width:1100px;margin:30px auto}canvas{background:white;border:1px solid #bbb}td,th{padding:9px;text-align:left}input{width:350px}.notice{background:#ffdfaf;padding:12px}</style>
<h1>觀測者選擇：電腦結果檢視</h1><p class="notice">合成資料只驗證程式流程；不代表 EgoExoMoCap 或真實資料的研究效果。姿態以關節點呈現，藍色為重建、橘色為參考。</p>
<table id="metrics"></table><p>策略 <select id="policy"></select> 幀 <input id="frame" type="range" min="0" value="0"> 視角 <input id="yaw" type="range" min="-180" max="180" value="20"></p>
<p id="info"></p><canvas width="1050" height="560" id="pose"></canvas><p>此頁嵌入完整 JSON：包含需求向量、來源集合、重訪與每幀誤差。灰色十字為原點；視角滑桿僅旋轉顯示。</p>
<script>const report=DATA;const names=Object.keys(report.policies);const select=document.querySelector('#policy');
const table=document.querySelector('#metrics');table.innerHTML='<tr><th>策略</th><th>MPJPE mm</th><th>單步 regret mm</th><th>來源請求次數</th><th>強制重訪名額</th><th>最長未更新秒數</th></tr>';
for(const n of names){let o=document.createElement('option');o.textContent=n;select.append(o);let r=report.policies[n];let tr=document.createElement('tr');for(const v of [n,r.mpjpe_mm?.toFixed(3)??'N/A',r.regret_mm?.toFixed(3)??'N/A',r.requests,r.refresh_requests,r.max_source_wait_s?.toFixed(3)??'N/A']){let td=document.createElement('td');td.textContent=v;tr.append(td)}table.append(tr)}
const slider=document.querySelector('#frame'),yaw=document.querySelector('#yaw'),canvas=document.querySelector('#pose'),ctx=canvas.getContext('2d');
function draw(){let frames=report.policies[select.value].frames;slider.max=Math.max(0,frames.length-1);let f=frames[Math.min(+slider.value,frames.length-1)];if(!f)return;document.querySelector('#info').textContent=`序列 ${f.sequence}，t=${f.timestamp.toFixed(3)}，選擇 ${f.selected_ids.join(' + ')}，重訪 ${f.required_refresh_ids.join(', ')||'無'}，${f.status}，MPJPE=${f.mpjpe_mm?.toFixed(2)??'N/A'} mm`;
ctx.clearRect(0,0,1050,560);if(!f.joints)return;let a=+yaw.value*Math.PI/180;let points=[...f.joints,...f.gt];let extent=Math.max(.2,...points.flatMap(p=>p.map(Math.abs)));let scale=240/extent;ctx.strokeStyle='#ccc';ctx.beginPath();ctx.moveTo(525,0);ctx.lineTo(525,560);ctx.moveTo(0,280);ctx.lineTo(1050,280);ctx.stroke();
for(const [arr,color] of [[f.gt,'#dd8c35'],[f.joints,'#3274b2']])arr.forEach((p,i)=>{let up=report.vertical_axis==='z'?2:1,depth=up===2?1:2;let x=525+scale*(p[0]*Math.cos(a)+p[depth]*Math.sin(a)),y=280+(report.vertical_direction==='down'?1:-1)*scale*p[up];ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,4,0,7);ctx.fill();ctx.fillText(i,x+6,y-4)})}
select.onchange=()=>{slider.value=0;draw()};slider.oninput=draw;yaw.oninput=draw;draw();</script></html>'''.replace('DATA',encoded,1)
    if report.get('experiment_scope') == 'training_only_no_validation_or_test':
        html=html.replace('合成資料只驗證程式流程；不代表 EgoExoMoCap 或真實資料的研究效果。',
                          'IN-SAMPLE：這是訓練過的同一段資料，沒有獨立驗證或測試，不能作為泛化成績。')
    Path(output).write_text(html,encoding='utf-8')


def audit(repo):
    repo = Path(repo)
    return {'repository':str(repo.resolve()),'repository_exists':repo.is_dir(),
        'entry_exists':(repo/'main_test.py').is_file(),
        'checkpoints':[str(p) for p in repo.rglob('*.pth')] if repo.is_dir() else [],
        'body_models':[str(p) for p in (repo/'third_party').rglob('*.npz')] if (repo/'third_party').exists() else [],
        'prepared_data':[str(p) for p in (repo/'dataset').glob('*')] if (repo/'dataset').exists() else [],
        'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'human_body_prior_available':importlib.util.find_spec('human_body_prior') is not None,
        'status':'Inventory only; this does not certify real model execution'}


def main():
    parser=argparse.ArgumentParser(description='Causal observer selection: frozen backend + expected MPJPE')
    subs=parser.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare-raw',help='Rectify official EgoHumans, initialize image identity, track and extract features')
    prep.add_argument('--sequence',required=True);prep.add_argument('--output',required=True)
    prep.add_argument('--target',default='aria01');prep.add_argument('--requester',default='aria02')
    prep.add_argument('--sources',nargs='+',default=['aria03','aria04'])
    prep.add_argument('--repo',default='external/EgoExoMoCap');prep.add_argument('--body-model')
    prep.add_argument('--device',default='cuda')
    prep.add_argument('--restart-setup',action='store_true',help='Retry pre-observation setup failure; preserve downloaded weights, reject existing observation outputs')
    fit=subs.add_parser('fit-sequence',help='Train both stages on one sequence; IN-SAMPLE results only')
    fit.add_argument('--data',required=True);fit.add_argument('--output',required=True)
    fit.add_argument('--body-model',required=True);fit.add_argument('--repo',default='external/EgoExoMoCap')
    fit.add_argument('--device',default='cuda');fit.add_argument('--recon-epochs',type=int,default=8)
    fit.add_argument('--selector-epochs',type=int,default=8)
    raw=subs.add_parser('inspect-egohumans',help='Read-only inventory of trusted official raw EgoHumans data')
    raw.add_argument('--sequence',required=True);raw.add_argument('--output',required=True)
    plots=subs.add_parser('report',help='Export run JSON to PNG/PDF/CSV')
    plots.add_argument('--run',required=True);plots.add_argument('--output',required=True)
    features=subs.add_parser('prepare-features',help='DINOv2/ViTPose extraction from prepared videos and audited detector boxes')
    features.add_argument('--pair',required=True);features.add_argument('--boxes',required=True)
    features.add_argument('--receipt',required=True);features.add_argument('--device',default='cuda')
    features.add_argument('--repo',default='external/EgoExoMoCap')
    features.add_argument('--vitpose',default='external/EgoExoMoCap/third_party/vitpose_hf/vitpose-base-simple')
    features.add_argument('--dinov2',default='external/EgoExoMoCap/third_party/dinov2_hf/dinov2-base')
    ap=subs.add_parser('audit');ap.add_argument('--repo',default='external/EgoExoMoCap');ap.add_argument('--output')
    check=subs.add_parser('check-data');check.add_argument('--protocol',required=True);check.add_argument('--output')
    export=subs.add_parser('export-egohumans');export.add_argument('--manifest',required=True);export.add_argument('--output',required=True)
    export.add_argument('--allow-benchmark-inputs',action='store_true')
    local_export=subs.add_parser('export-local-egohumans')
    local_export.add_argument('--manifest',required=True);local_export.add_argument('--output',required=True)
    recon=subs.add_parser('train-reconstructor')
    recon.add_argument('--protocol',required=True);recon.add_argument('--output',required=True)
    recon.add_argument('--repo',default='external/EgoExoMoCap');recon.add_argument('--body-model',required=True)
    recon.add_argument('--device',default='cpu');recon.add_argument('--epochs',type=int,default=8)
    recon.add_argument('--window-size',type=int,default=16);recon.add_argument('--embed-dim',type=int,default=128)
    recon.add_argument('--max-sources',type=int,default=2);recon.add_argument('--seed',type=int,default=7)
    recon.add_argument('--lr',type=float,default=1e-4)
    demo=subs.add_parser('demo');demo.add_argument('--output',required=True);demo.add_argument('--epochs',type=int,default=8)
    demo.add_argument('--memory',choices=['gru','attention'],default='gru');demo.add_argument('--seed',type=int,default=7)
    acquisition_args(demo)
    for name in ('train','evaluate'):
        p=subs.add_parser(name);p.add_argument('--data');p.add_argument('--protocol');p.add_argument('--output',required=True)
        p.add_argument('--backend',choices=['weighted','egoexo','local-aux'],default='local-aux')
        p.add_argument('--repo',default='external/EgoExoMoCap');p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--body-model')
        p.add_argument('--device',default='cpu');p.add_argument('--allow-benchmark-inputs',action='store_true')
        if name=='train':
            acquisition_args(p)
            p.add_argument('--epochs',type=int,default=8);p.add_argument('--validation-data')
            p.add_argument('--memory',choices=['gru','attention'],default='gru');p.add_argument('--seed',type=int,default=7)
        else: p.add_argument('--model',required=True)
    args=parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(),4))
    if args.command=='prepare-raw':
        from .raw_prepare import prepare_raw
        print(json.dumps(prepare_raw(args.sequence,args.output,args.target,args.requester,args.sources,
            args.repo,args.body_model,args.device,args.restart_setup),indent=2));return
    if args.command=='fit-sequence':
        from .single_sequence import fit_sequence
        print(json.dumps(fit_sequence(args.data,args.output,args.repo,args.body_model,args.device,
            args.recon_epochs,args.selector_epochs),indent=2));return
    if args.command=='inspect-egohumans':
        from .raw_inventory import inventory_raw
        print(json.dumps(inventory_raw(args.sequence,args.output),indent=2,ensure_ascii=True));return
    if args.command=='report':
        from .reports import export_report
        print(json.dumps(export_report(args.run,args.output),indent=2));return
    if args.command=='prepare-features':
        from .image_features import prepare_dinov2_features
        prepare_dinov2_features(args.pair,args.boxes,args.receipt,args.vitpose,args.dinov2,args.repo,args.device)
        print('DINOv2/ViTPose features complete');return
    if args.command=='export-local-egohumans':
        from .local_data import export_local_egohumans
        print(json.dumps(export_local_egohumans(args.manifest,args.output),ensure_ascii=False,indent=2));return
    if args.command=='train-reconstructor':
        from .local_fusion import LocalFusionConfig
        from .local_training import train_local_reconstructor
        result=train_local_reconstructor(args.protocol,args.output,args.repo,args.body_model,
            LocalFusionConfig(window_size=args.window_size,embed_dim=args.embed_dim),
            epochs=args.epochs,lr=args.lr,max_sources=args.max_sources,device=args.device,seed=args.seed)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if args.command=='export-egohumans':
        from .egohumans import export_prepared_egohumans
        print(json.dumps(export_prepared_egohumans(args.manifest,args.output,args.allow_benchmark_inputs),ensure_ascii=False,indent=2));return
    if args.command=='check-data':
        _,manifest=load_protocol(args.protocol)
        if args.output: _save_json(args.output,manifest)
        print(json.dumps(manifest,ensure_ascii=False,indent=2));return
    if args.command=='audit':
        result=audit(args.repo)
        if args.output: _save_json(args.output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if args.command=='demo':
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        if (out/'selector.pt').exists(): raise FileExistsError('Choose a new output directory')
        for name,seed in [('train',args.seed),('validation',args.seed+1),('test',args.seed+2)]:
            make_synthetic(out/f'{name}.npz',seed=seed)
        backend=WeightedJointBackend()
        network,_=train(out/'train.npz',out,backend,TrainConfig(epochs=args.epochs,seed=args.seed,memory_type=args.memory,
            acquisition=acquisition_from_args(args)),out/'validation.npz')
        network=load_network(out/'best.pt')
        verify_heldout(network,[out/'test.npz'],backend)
        report=evaluate(network,[load_sequence(out/'test.npz')],backend)
        report['dataset']='synthetic_demo: held-out seed, not real research evidence'
    else:
        if args.protocol:
            if args.data or getattr(args,'validation_data',None):
                raise ValueError('--protocol is exclusive with --data/--validation-data')
            parts,protocol_manifest=load_protocol(args.protocol)
            data=parts['train'] if args.command=='train' else parts['test']
            validation=parts['validation']
        else:
            if not args.data: raise ValueError('Specify --protocol for real data, or --data for synthetic fixtures')
            data=sequence_paths(args.data)
            protocol_manifest=audit_partitions({'data':data})
            if any(not r['is_synthetic'] for r in protocol_manifest['data']):
                raise ValueError('Real research runs require explicit --protocol with scene-disjoint train/validation/test')
            validation=getattr(args,'validation_data',None)
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        if args.command=='evaluate' and (out/'evaluation.json').exists():
            raise FileExistsError('Refusing to overwrite an existing evaluation; choose a fresh output')
        backend=backend_from_args(args)
        if args.backend=='local-aux':
            from .local_training import check_backend_splits
            check_backend_splits(backend,protocol_manifest)
        _save_json(out/'protocol_audit.json',protocol_manifest)
        if args.command=='train':
            train(data,out,backend,TrainConfig(epochs=args.epochs,device=args.device,seed=args.seed,memory_type=args.memory,
                acquisition=acquisition_from_args(args)),validation,args.allow_benchmark_inputs)
            return
        network=load_network(args.model,args.device)
        records=verify_heldout(network,data,backend)
        sequences=ReplaySequences(data,args.allow_benchmark_inputs)
        report=evaluate(network,sequences,backend)
        report['heldout_data_manifest']=records
        report['backend_provenance']=network.backend_provenance
    _save_json(out/'evaluation.json',report);write_viewer(report,out/'viewer.html')
    print(f'Results: {out.resolve()/"viewer.html"}',flush=True)
