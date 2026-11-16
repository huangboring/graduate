"""Explicit in-sample training; never relabel a single sequence as held-out."""
from pathlib import Path
import json
from .local_training import train_local_reconstructor
from .local_fusion import LocalFusionConfig, LocalAuxBackend
from .training import TrainConfig, train, evaluate, load_network, _save_json
from .replay import load_sequence
from .selection import AcquisitionConfig
from .reports import export_report


def fit_sequence(data, output, repo, body_model, device='cuda', recon_epochs=8, selector_epochs=8):
    from .preparation_quality import check_preparation
    quality=check_preparation(data)
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a fresh experiment directory')
    out.mkdir(parents=True,exist_ok=True)
    _save_json(out/'preparation_quality.json',quality)
    _save_json(out/'experiment.json',{'scope':'training_only_no_validation_or_test',
        'data':str(Path(data).resolve()),'max_sources':1,
        'warning':'Every curve/pose/policy comparison is in-sample. No generalization claim.'})
    reconstruction=train_local_reconstructor(None,out/'reconstructor',repo,body_model,
        LocalFusionConfig(),epochs=recon_epochs,max_sources=1,device=device,train_only_data=data)
    export_report(out/'reconstructor',out/'reconstructor/figures')
    backend=LocalAuxBackend.from_checkpoint(repo,reconstruction['checkpoint'],body_model,device)
    train(data,out/'selector',backend,TrainConfig(epochs=selector_epochs,device=device,training_only=True,
        acquisition=AcquisitionConfig(max_sources=1)))
    export_report(out/'selector',out/'selector/figures')
    network=load_network(out/'selector/selector.pt',device)
    print('Generating explicitly IN-SAMPLE policy comparisons and pose viewer...',flush=True)
    report=evaluate(network,[load_sequence(data)],backend,['learned','random','round_robin','all'])
    report['experiment_scope']='training_only_no_validation_or_test'
    report['dataset']='EgoHumans training sequence: IN-SAMPLE, not a test result'
    result=out/'training_results';result.mkdir()
    _save_json(result/'evaluation.json',report)
    from .cli import write_viewer
    write_viewer(report,result/'viewer.html')
    export_report(result,result/'figures')
    return {'scope':report['experiment_scope'],'output':str(out.resolve()),
            'viewer':str((result/'viewer.html').resolve()),'heldout_results':False}
