import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_dataloaders, load_splits_from_csv
from model import build_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        probs = torch.softmax(out, dim=1)

        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(cm, class_names, outputs_dir):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    out = Path(outputs_dir) / 'confusion_matrix.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out}")


def evaluate(data_dir='data', outputs_dir='outputs',
             checkpoint='outputs/best_model.pth', batch_size=64, num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_p, train_l, val_p, val_l, test_p, test_l, class_names = load_splits_from_csv(data_dir)
    num_classes = len(class_names)

    _, _, test_loader = get_dataloaders(
        train_p, train_l, val_p, val_l, test_p, test_l,
        batch_size=batch_size, num_workers=num_workers,
    )

    model = build_model(num_classes, freeze_backbone=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])

    preds, labels, probs = get_predictions(model, test_loader, device)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec  = recall_score(labels, preds, average='macro', zero_division=0)
    f1   = f1_score(labels, preds, average='macro', zero_division=0)

    try:
        roc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        roc = float('nan')

    print('\n=== Test Set Results ===')
    print(f'  Accuracy   : {acc:.4f}')
    print(f'  Precision  : {prec:.4f}  (macro)')
    print(f'  Recall     : {rec:.4f}  (macro)')
    print(f'  F1-score   : {f1:.4f}  (macro)')
    print(f'  ROC-AUC    : {roc:.4f}  (one-vs-rest macro)')
    print('\nPer-class breakdown:')
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, outputs_dir)

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='data')
    parser.add_argument('--outputs_dir', default='outputs')
    parser.add_argument('--checkpoint',  default='outputs/best_model.pth')
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
