"""
Phase 6 model comparison: build a table from several training runs.

Each run directory is an --outputs_dir that went through train.py and
evaluate.py, so it holds train_summary.json and metrics.json.

Usage:
    python src/compare_models.py outputs/resnet18 outputs/efficientnet_b0
    python src/compare_models.py outputs/resnet18 outputs/resnet34 --out outputs/model_comparison.md
"""

import argparse
import json
from pathlib import Path


def load_run(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / 'metrics.json') as f:
        metrics = json.load(f)
    summary_path = run_dir / 'train_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return metrics, summary


def fmt_time(sec):
    if sec is None:
        return 'n/a'
    return f'{sec / 60:.1f} min'


def build_table(run_dirs):
    header = ('| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) '
              '| ROC-AUC | Params | Size (MB) | Train time | Best stage |')
    sep = '|' + '---|' * 10
    rows = [header, sep]

    for run_dir in run_dirs:
        m, s = load_run(run_dir)
        roc = m.get('roc_auc_ovr_macro')
        best = s.get('best_checkpoint', {})
        rows.append(
            f"| {m['arch']} "
            f"| {m['accuracy']:.4f} "
            f"| {m['precision_macro']:.4f} "
            f"| {m['recall_macro']:.4f} "
            f"| {m['f1_macro']:.4f} "
            f"| {'n/a' if roc is None else f'{roc:.4f}'} "
            f"| {m['total_params'] / 1e6:.2f}M "
            f"| {m['model_size_mb']:.1f} "
            f"| {fmt_time(s.get('train_time_sec'))} "
            f"| {best.get('stage', 'n/a')} ep {best.get('epoch', '-')} |"
        )
    return '\n'.join(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='+', help='One outputs dir per model')
    parser.add_argument('--out', default='outputs/model_comparison.md')
    args = parser.parse_args()

    table = build_table(args.run_dirs)
    print(table)
    Path(args.out).write_text('# Model Comparison (test set)\n\n' + table + '\n')
    print(f'\nSaved to {args.out}')
