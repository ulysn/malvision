import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_transforms, load_splits_from_csv
from model import build_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def tensor_to_rgb(tensor):
    img = tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1).astype(np.float32)


def gradcam_for_sample(model, tensor, class_idx, device):
    target_layer = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale = cam(
            input_tensor=tensor.unsqueeze(0).to(device),
            targets=[ClassifierOutputTarget(class_idx)],
        )
    return grayscale[0]


def generate_all(data_dir='data', outputs_dir='outputs',
                 checkpoint='outputs/best_model.pth', samples_per_class=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, _, _, test_paths, test_labels, class_names = load_splits_from_csv(data_dir)
    num_classes = len(class_names)

    model = build_model(num_classes, freeze_backbone=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    transform = get_transforms(augment=False)

    # Collect up to N samples per class from the test set
    class_samples: dict[int, list] = {i: [] for i in range(num_classes)}
    for path, label in zip(test_paths, test_labels):
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(path)

    for class_idx, class_name in enumerate(class_names):
        out_dir = Path(outputs_dir) / 'gradcam' / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = class_samples[class_idx]
        print(f"Grad-CAM  {class_name}: {len(paths)} samples")

        for i, img_path in enumerate(paths):
            tensor = transform(Image.open(img_path).convert('L'))
            display = tensor_to_rgb(tensor)

            cam_mask = gradcam_for_sample(model, tensor, class_idx, device)
            overlay  = show_cam_on_image(display, cam_mask, use_rgb=True)

            out_path = out_dir / f'sample_{i + 1}.png'
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"  Saved to {out_dir}")

    print("Grad-CAM generation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',          default='data')
    parser.add_argument('--outputs_dir',        default='outputs')
    parser.add_argument('--checkpoint',         default='outputs/best_model.pth')
    parser.add_argument('--samples_per_class',  type=int, default=4)
    args = parser.parse_args()

    generate_all(
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        checkpoint=args.checkpoint,
        samples_per_class=args.samples_per_class,
    )
