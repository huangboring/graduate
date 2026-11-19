"""Summarize scorer/epoch/camera ablations without choosing on test data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
    root=Path(root)
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for scorer in ('mlp','attention'):
        run=root/scorer
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
                    rows.append({'scorer':scorer,'epoch':epoch,'split':split,'method':method,**values})
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
    print(f'Wrote {root}/comparison.csv and comparison.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    report(parser.parse_args().root)
