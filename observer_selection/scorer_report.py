"""Summarize scorer/epoch/camera ablations without choosing on test data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root, baseline_root=None):
    root=Path(root)
    rows=[]
    architectures=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for scorer in ('mlp','attention'):
        run=root/scorer
        contract=json.loads((run/'experiment.json').read_text(encoding='utf-8'))
        encoding=contract.get('encoding','legacy')
        architecture=run/'model_architecture.json'
        if architecture.exists():
            architectures.append(json.loads(architecture.read_text(encoding='utf-8')))
        history=json.loads((run/'learning_curve.json').read_text(encoding='utf-8'))
        axes[0].plot([r['epoch'] for r in history],[r['mpjpe_mm'] for r in history],label=scorer)
        axes[1].plot([r['epoch'] for r in history],[r['mean_requests'] for r in history],label=scorer)
        for path in sorted(run.glob('epoch_*/metrics.json')):
            epoch=int(path.parent.name.split('_')[1])
            for split,file in [('train12',path),('transfer34',path.parent/'transfer34/camera_transfer.json')]:
                if not file.exists():
                    continue
                metrics=json.loads(file.read_text(encoding='utf-8'))
                for method,values in metrics['methods'].items():
                    rows.append({'encoding':encoding,'scorer':scorer,'epoch':epoch,'split':split,'method':method,**values})
    if not rows:
        raise ValueError('No milestone results found')
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for scorer in ('mlp','attention'):
        for split in ('train12','transfer34'):
            points=[r for r in rows if r['scorer']==scorer and r['split']==split and r['method']=='Learned']
            if points:
                axes[2].plot([r['epoch'] for r in points],[r['mpjpe_mm'] for r in points],marker='o',label=f'{scorer} {split}')
    for ax,title,ylabel in zip(axes,['Training-sequence error','Requests per frame','Fixed-weight camera transfer'],['MPJPE (mm)','Requests','MPJPE (mm)']):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same sequence; cam01/02 training vs cam03/04 transfer; aria03/04 shared',fontsize=11)
    fig.savefig(root/'comparison.png',dpi=180)
    plt.close(fig)
    if architectures:
        with (root/'model_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(architectures[0]))
            writer.writeheader();writer.writerows(architectures)
    if baseline_root is not None:
        compare_baseline(root,Path(baseline_root),rows)
    print(f'Wrote {root}/comparison.csv and comparison.png')


def compare_baseline(root, baseline, rows):
    from .preparation_cache import atomic_json
    comparable=[]
    status=[]
    # Encoding/head changes are intentional. All data, teacher, cost and cameras must match.
    keys=('replay_sha256','sources','checkpoint_sha256','sequence','windows','frozen_sha256',
          'objective','label_stride','target_context','entropy_temperature_m','cost_m_per_source')
    for scorer in ('mlp','attention'):
        old=baseline/scorer
        file=old/'experiment.json'
        if not file.exists():
            status.append({'scorer':scorer,'matched':False,'reason':'Baseline run absent'})
            continue
        before=json.loads(file.read_text(encoding='utf-8'))
        after=json.loads((root/scorer/'experiment.json').read_text(encoding='utf-8'))
        differences=[key for key in keys if before.get(key)!=after.get(key)]
        if differences:
            status.append({'scorer':scorer,'matched':False,'different_fields':differences})
            continue
        used=0
        for row in rows:
            if row['scorer']!=scorer:
                continue
            name='metrics.json' if row['split']=='train12' else 'transfer34/camera_transfer.json'
            epoch=old/f"epoch_{row['epoch']:04}"
            source=epoch/name
            if not source.exists():
                continue
            # Check the exact baseline milestone, not only the latest run contract.
            milestone=json.loads((epoch/'experiment.json').read_text(encoding='utf-8'))
            if any(milestone.get(key)!=after.get(key) for key in keys):
                continue
            if row['split']=='transfer34':
                old_protocol=json.loads((source.parent/'evaluation_protocol.json').read_text(encoding='utf-8'))
                new_protocol=json.loads((root/scorer/f"epoch_{row['epoch']:04}"/'transfer34/evaluation_protocol.json').read_text(encoding='utf-8'))
                if (old_protocol['evaluation_sources']!=new_protocol['evaluation_sources'] or
                    not old_protocol.get('evaluation_replay_sha256') or
                    old_protocol['evaluation_replay_sha256']!=new_protocol.get('evaluation_replay_sha256')):
                    continue
            value=json.loads(source.read_text(encoding='utf-8'))['methods'].get(row['method'])
            if value is None:
                continue
            comparable.append({'scorer':scorer,'epoch':row['epoch'],'split':row['split'],'method':row['method'],
                'old_encoding':before.get('encoding','legacy'),'new_encoding':row['encoding'],
                'old_mpjpe_mm':value['mpjpe_mm'],'new_mpjpe_mm':row['mpjpe_mm'],
                'delta_mpjpe_mm':row['mpjpe_mm']-value['mpjpe_mm'],
                'old_mean_requests':value['mean_requests'],'new_mean_requests':row['mean_requests'],
                'delta_mean_requests':row['mean_requests']-value['mean_requests']})
            used+=1
        status.append({'scorer':scorer,'matched':bool(used),'matched_rows':used})
    atomic_json(root/'baseline_comparison_status.json',{
        'baseline':str(baseline),'status':status,
        'scope':'Same-sequence exploratory ablation; negative error delta is better. Changes include feature encoding AND scoring-head design.'})
    if not comparable:
        for name in ('old_vs_new.csv','old_vs_new.png'):
            (root/name).unlink(missing_ok=True)  # Remove only this report's stale generated outputs.
        print('No matched legacy/new milestones; see baseline_comparison_status.json')
        return
    with (root/'old_vs_new.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparable[0]))
        writer.writeheader();writer.writerows(comparable)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,split in zip(axes,('train12','transfer34')):
        for scorer in ('mlp','attention'):
            values=sorted([r for r in comparable if r['split']==split and r['scorer']==scorer and r['method']=='Learned'],key=lambda r:r['epoch'])
            for kind,style in [('old','--'),('new','-')]:
                if values:
                    ax.plot([r['epoch'] for r in values],[r[kind+'_mpjpe_mm'] for r in values],style,marker='o',label=f'{scorer} {kind}')
        ax.set(title=split,xlabel='Epoch',ylabel='MPJPE (mm)');ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle('Matched frozen reconstruction and data; same-sequence evaluation',fontsize=11)
    fig.savefig(root/'old_vs_new.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--baseline-root')
    args=parser.parse_args()
    report(args.root,args.baseline_root)
