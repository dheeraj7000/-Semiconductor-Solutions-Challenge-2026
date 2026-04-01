"""
Final Inference Script — 9-Class Output

Outputs one final class:
    good
    defect1 ... defect9
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MODEL_PATH = "reports/resnet50_multitask/best_multitask.pt"
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEFECT_CLASSES = [
    'defect1', 'defect10', 'defect2',
    'defect3', 'defect4', 'defect5',
    'defect8', 'defect9'
]

# ------------------------------------------------------------
# Sobel preprocessing (same as training)
# ------------------------------------------------------------
_SOBEL_X = ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1, offset=128)
_SOBEL_Y = ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1, offset=128)

def grayscale_to_edge_channels(img_gray):
    sobel_x = img_gray.filter(_SOBEL_X)
    sobel_y = img_gray.filter(_SOBEL_Y)
    return Image.merge("RGB", [img_gray, sobel_x, sobel_y])

# ------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------
class MultiTaskResNet50(nn.Module):
    def __init__(self, num_defect_classes):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.binary_head = nn.Linear(2048, 1)
        self.subtype_head = nn.Linear(2048, num_defect_classes)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        bin_logit = self.binary_head(feat).squeeze(1)
        sub_logits = self.subtype_head(feat)
        return bin_logit, sub_logits

# ------------------------------------------------------------
# Transform
# ------------------------------------------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ------------------------------------------------------------
# TTA
# ------------------------------------------------------------
def apply_tta(model, x):
    model.eval()
    with torch.no_grad():
        b1, s1 = model(x)

        x_h = torch.flip(x, dims=[3])
        b2, s2 = model(x_h)

        x_v = torch.flip(x, dims=[2])
        b3, s3 = model(x_v)

        bin_logit = (b1 + b2 + b3) / 3
        sub_logits = (s1 + s2 + s3) / 3

    return bin_logit, sub_logits

# ------------------------------------------------------------
# Final Prediction Logic (9-class output)
# ------------------------------------------------------------
def predict_image(model, image_path):
    img = Image.open(image_path).convert("L")
    img_rgb = grayscale_to_edge_channels(img)
    x = transform(img_rgb).unsqueeze(0).to(DEVICE)

    bin_logit, sub_logits = apply_tta(model, x)

    # Binary decision — compare raw logit against tuned threshold
    bin_logit_val = bin_logit.item()
    is_defect = bin_logit_val > -0.152  # best_threshold from training sweep

    if not is_defect:
        return "good"

    # Defect subtype decision
    sub_probs = torch.softmax(sub_logits, dim=1)
    sub_idx = torch.argmax(sub_probs, dim=1).item()
    return DEFECT_CLASSES[sub_idx]

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    model = MultiTaskResNet50(num_defect_classes=len(DEFECT_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    input_path = Path(args.input)

    if input_path.is_file():
        final_class = predict_image(model, input_path)
        print(f"{input_path.name} -> {final_class}")

    elif input_path.is_dir():
        print(f"Running inference on folder: {input_path}")
        for img_file in input_path.glob("*.*"):
            if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                final_class = predict_image(model, img_file)
                print(f"{img_file.name} -> {final_class}")

    else:
        print("Invalid input path.")

if __name__ == "__main__":
    main()
