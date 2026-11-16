"""Export measured run records to PNG/PDF and CSV. Never invent missing metrics."""
import csv
import json
from pathlib import Path
import numpy as np


def export_report(run_dir, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .protocol import sha256_file
    run, out = Path(run_dir), Path(output)
    inputs = [p for p in (run/'metrics.json', run/'training.json', run/'evaluation.json') if p.is_file()]
    if not inputs:
        raise FileNotFoundError('Need metrics.json, training.json or evaluation.json from a completed run')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh report directory')
    out.mkdir(parents=True, exist_ok=True)
    files = []

    def save(fig, name):
        for ext in ('png', 'pdf'):
            target = out/f'{name}.{ext}'
            fig.savefig(target, dpi=300, bbox_inches='tight')
            files.append(target.name)
        plt.close(fig)

    def table(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for row in rows for k in row))
        path = out/f'{name}.csv'
        with path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
        files.append(path.name)

    for path in inputs:
        data = json.loads(path.read_text(encoding='utf-8'))
        if path.name != 'evaluation.json':
            table(path.stem, data)
            if not data:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            recon = path.name == 'metrics.json'
            fields = [('train_mpjpe_mm', 'Train MPJPE', 1),
                      ('validation_scene_macro_mpjpe_mm', 'Validation scene macro MPJPE', 1)] if recon else [
                      ('expected_mpjpe_m', 'Train expected MPJPE objective', 1000),
                      ('rollin_mpjpe_mm', 'Train sampled-policy MPJPE', 1),
                      ('validation_mpjpe_mm', 'Validation scene macro MPJPE', 1)]
            for key, label, scale in fields:
                rows = [r for r in data if r.get(key) is not None]
                if rows:
                    ax.plot([r['epoch'] + int(recon) for r in rows], [r[key]*scale for r in rows], label=label)
            ax.set(xlabel='Epoch', ylabel='Error / objective (mm)',
                   title='Reconstructor training' if recon else 'Observer selector training')
            ax.grid(alpha=.2); ax.legend(fontsize=8); save(fig, path.stem+'_curves')
            continue

        policies = data['policies']
        synthetic = ('synthetic_demo' in str(data.get('dataset', '')) or
                     any(r.get('is_synthetic', False) for r in data.get('heldout_data_manifest', [])))
        notice = 'SYNTHETIC: pipeline verification only' if synthetic else 'Evaluation: verify split provenance in JSON'
        summaries = []
        for name, result in policies.items():
            summaries.append({'policy': name, 'scene_macro_mpjpe_mm': result.get('scene_macro_mpjpe_mm'),
                'frame_mean_mpjpe_mm': result.get('mpjpe_mm'), 'requests': result.get('requests'),
                'refresh_requests': result.get('refresh_requests'),
                'evaluated_frames': result.get('evaluated_frames'), 'budget_note': result.get('budget_note')})
        table('policy_summary', summaries)
        rows = [r for r in summaries if r['scene_macro_mpjpe_mm'] is not None]
        if rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar([r['policy'] for r in rows], [r['scene_macro_mpjpe_mm'] for r in rows])
            for bar, row in zip(bars, rows):
                if row['policy'] == 'all':
                    bar.set_hatch('//'); bar.set_color('#aaaaaa')
            ax.set(ylabel='Scene macro MPJPE (mm)', title='Policy comparison (all = different acquisition budget)')
            fig.text(.01, -.03, notice, fontsize=8); save(fig, 'policy_comparison')

        sequence_ids = sorted({r['sequence'] for p in policies.values() for r in p['frames']})
        for seq_id in sequence_ids:
            fig, ax = plt.subplots(figsize=(9, 4)); any_error = False
            for name, result in policies.items():
                rows = [r for r in result['frames'] if r['sequence'] == seq_id and r.get('mpjpe_mm') is not None]
                table(f'sequence_{seq_id}_{name}', [{k: r.get(k) for k in ('timestamp','mpjpe_mm','oracle_mpjpe_mm','selected_ids','required_refresh_ids','status')} for r in rows])
                if rows:
                    any_error = True
                    ax.plot([r['timestamp'] for r in rows], [r['mpjpe_mm'] for r in rows], label=name, alpha=.8)
            ax.set(xlabel='Time (s)', ylabel='MPJPE (mm)', title=f'Sequence {seq_id}: unsmoothed errors')
            if any_error: ax.legend(fontsize=8)
            fig.text(.01, -.03, notice, fontsize=8); save(fig, f'sequence_{seq_id}_errors')
            learned = policies.get('learned', {})
            rows = [r for r in learned.get('frames', []) if r['sequence'] == seq_id]
            coverage = learned.get('source_coverage', {}).get(str(seq_id), {}).get('acquisition_counts', {})
            sources = sorted(set(coverage) | {s for r in rows for s in r['selected_ids']})
            if rows and sources:
                matrix = np.array([[int(s in r['selected_ids']) + int(s in r.get('required_refresh_ids', []))
                                    for r in rows] for s in sources])
                fig, ax = plt.subplots(figsize=(9, max(2.5, len(sources)*.4)))
                from matplotlib.colors import ListedColormap, BoundaryNorm
                colors = ListedColormap(['#eeeeee', '#3976aa', '#e29642'])
                im = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=colors,
                               norm=BoundaryNorm([-.5,.5,1.5,2.5], 3))
                ax.set(yticks=range(len(sources)), yticklabels=sources, xlabel='Frame index (includes warmup)', title=f'Sequence {seq_id}: acquired observers')
                cb = fig.colorbar(im, ax=ax, ticks=[0,1,2]); cb.ax.set_yticklabels(['Not acquired','Selected','Mandatory refresh'])
                fig.text(.01, -.03, notice, fontsize=8); save(fig, f'sequence_{seq_id}_selection')

    manifest = {'inputs': [{'path': str(p.resolve()), 'sha256': sha256_file(p)} for p in inputs],
                'files': files, 'smoothing': False,
                'notes': ['No confidence intervals from a single run.',
                          'All-view acquisition has a different budget; it is not guaranteed to be a lower error bound.',
                          'Training curves do not establish held-out generalization.',
                          'Warmup frames without reconstruction are excluded from error curves.']}
    (out/'report_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
