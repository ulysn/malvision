import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (create_splits, get_class_weights, get_dataloaders,
                     list_families, load_splits_from_csv)
from model import (SUPPORTED_ARCHS, build_model, count_params, count_trainable,
                   unfreeze_all, unfreeze_last_blocks)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = criterion(out, labels)

        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def run_stage(model, train_loader, val_loader, criterion, device,
              lr, epochs, stage_name, outputs_dir, history, best, ckpt_meta):
    # `best` is shared across stages so best_model.pth is the global best,
    # not just the best of whichever stage ran last.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_path = Path(outputs_dir) / 'best_model.pth'

    print(f"\n{'='*60}")
    print(f"  {stage_name}   trainable params: {count_trainable(model):,}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(
            ('stage', 'train_loss', 'train_acc', 'val_loss', 'val_acc'),
            (stage_name, tr_loss, tr_acc, va_loss, va_acc),
        ):
            history[k].append(v)

        marker = ''
        if va_loss < best['val_loss']:
            best.update(val_loss=va_loss, val_acc=va_acc, stage=stage_name, epoch=epoch)
            torch.save({'model_state': model.state_dict(), 'stage': stage_name,
                        'epoch': epoch, **ckpt_meta}, best_path)
            marker = '  <- best'

        print(
            f"  epoch {epoch:3d}/{epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}{marker}"
        )

    return history


def plot_history(history, outputs_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='train')
    ax1.plot(history['val_loss'],   label='val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='train')
    ax2.plot(history['val_acc'],   label='val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out = Path(outputs_dir) / 'training_curves.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training curves saved to {out}")


def load_or_create_splits(data_dir, processed_dir, rebuild=False):
    """Reuse the saved split CSVs unless they were built from a different set of
    families than what is in processed_dir now (e.g. Malimg CSVs left over when
    switching to the 7 BIG 2015 families). In that case, rebuild them."""
    if not rebuild:
        try:
            splits = load_splits_from_csv(data_dir)
        except FileNotFoundError:
            pass
        else:
            if splits[-1] == list_families(processed_dir):
                return splits
            print(f"Saved splits in {data_dir}/ do not match the families in "
                  f"{processed_dir}/, rebuilding them.")
    return create_splits(processed_dir, data_dir)


def train(data_dir='data', processed_dir='data/processed', outputs_dir='outputs',
          batch_size=64, num_workers=4, run_stage3=False, arch='resnet18',
          rebuild_splits=False, epochs=(12, 12, 8)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}   arch: {arch}")
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    splits = load_or_create_splits(data_dir, processed_dir, rebuild=rebuild_splits)

    train_p, train_l, val_p, val_l, test_p, test_l, class_names = splits
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    train_loader, val_loader, _ = get_dataloaders(
        train_p, train_l, val_p, val_l, test_p, test_l,
        batch_size=batch_size, num_workers=num_workers,
    )

    class_weights = get_class_weights(train_l, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = build_model(num_classes, freeze_backbone=True, arch=arch).to(device)
    history = {'stage': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best = {'val_loss': float('inf')}
    ckpt_meta = {'arch': arch, 'class_names': class_names}
    stage_kwargs = dict(outputs_dir=outputs_dir, history=history, best=best, ckpt_meta=ckpt_meta)
    start = time.time()

    run_stage(model, train_loader, val_loader, criterion, device,
              lr=1e-3, epochs=epochs[0], stage_name='Stage1', **stage_kwargs)

    unfreeze_last_blocks(model, arch)
    run_stage(model, train_loader, val_loader, criterion, device,
              lr=1e-4, epochs=epochs[1], stage_name='Stage2', **stage_kwargs)

    if run_stage3:
        unfreeze_all(model)
        run_stage(model, train_loader, val_loader, criterion, device,
                  lr=1e-5, epochs=epochs[2], stage_name='Stage3', **stage_kwargs)

    train_time = time.time() - start
    plot_history(history, outputs_dir)

    summary = {
        'arch': arch,
        'num_classes': num_classes,
        'class_names': class_names,
        'total_params': count_params(model),
        'train_time_sec': round(train_time, 1),
        'best_checkpoint': {k: best[k] for k in ('stage', 'epoch', 'val_loss', 'val_acc')},
        'history': history,
    }
    with open(Path(outputs_dir) / 'train_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete in {train_time / 60:.1f} min. "
          f"Best: {best['stage']} epoch {best['epoch']} (val loss {best['val_loss']:.4f})")
    return model, class_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',      default='data')
    parser.add_argument('--processed_dir', default='data/processed')
    parser.add_argument('--outputs_dir',   default='outputs')
    parser.add_argument('--batch_size',    type=int, default=64)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--stage3',        action='store_true')
    parser.add_argument('--arch',          default='resnet18', choices=SUPPORTED_ARCHS)
    parser.add_argument('--rebuild_splits', action='store_true',
                        help='Ignore saved split CSVs and re-split processed_dir')
    parser.add_argument('--epochs', type=int, nargs=3, default=[12, 12, 8],
                        metavar=('S1', 'S2', 'S3'), help='Epochs per stage')
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        outputs_dir=args.outputs_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        run_stage3=args.stage3,
        arch=args.arch,
        rebuild_splits=args.rebuild_splits,
        epochs=tuple(args.epochs),
    )
