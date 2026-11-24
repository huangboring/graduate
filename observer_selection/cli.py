"""Data preparation commands. Training: python -m observer_selection.fit_triangulated."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare-raw', help='Prepare images, target identity, poses and camera metadata')
    prep.add_argument('--sequence', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--target', default='aria01')
    prep.add_argument('--requester', default='aria02')
    prep.add_argument('--sources', nargs='+', default=['aria03', 'aria04'])
    prep.add_argument('--repo', default='external/EgoExoMoCap')
    prep.add_argument('--body-model')
    prep.add_argument('--device', default='cuda')
    for flag in ('restart-setup', 'resume', 'review-tracking', 'annotation-boxes'):
        prep.add_argument('--'+flag, action='store_true')
    prep.add_argument('--model-cache')
    prep.add_argument('--reset-view', nargs='+', default=[])
    prep.add_argument('--reuse-candidates')
    final = subs.add_parser('finalize-raw', help='Validate completed preparation without GPU rerun')
    final.add_argument('--prepared', required=True)
    final.add_argument('--output', required=True)
    identity = subs.add_parser('audit-identity', help='Read-only target ID annotation gallery')
    identity.add_argument('--sequence', required=True)
    identity.add_argument('--output', required=True)
    identity.add_argument('--frame', type=int, default=1)
    inventory = subs.add_parser('inspect-egohumans', help='Read-only raw dataset inventory')
    inventory.add_argument('--sequence', required=True)
    inventory.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare-raw':
        from .raw_prepare import prepare_raw
        result = prepare_raw(args.sequence, args.output, args.target, args.requester, args.sources,
            args.repo, args.body_model, args.device, args.restart_setup, args.resume,
            args.model_cache, args.reset_view, args.review_tracking, args.reuse_candidates,
            args.annotation_boxes)
    elif args.command == 'finalize-raw':
        from .finalize_raw import finalize_raw
        result = finalize_raw(args.prepared, args.output)
    elif args.command == 'audit-identity':
        from .identity_audit import audit_identity
        result = audit_identity(args.sequence, args.output, args.frame)
    else:
        from .raw_inventory import inventory_raw
        result = inventory_raw(args.sequence, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
