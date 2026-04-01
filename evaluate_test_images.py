"""
Evaluate fine-tuned MultiTask ResNet-50 on Intel test images.
Produces:
    - Overall accuracy
    - Per-class accuracy, precision, recall, F1
    - Confusion matrix (heatmap)
    - Per-class bar charts
    - JSON summary
"""

import os, json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

# ---- reuse model & predict from inference script ----
from inference_resnet50_multitask import (
    MultiTaskResNet50, predict_image, DEFECT_CLASSES, MODEL_PATH, DEVICE
)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
TEST_DIR = Path("inteltestimages")
OUT_DIR = Path("reports/test_evaluation")
ALL_CLASSES = ["good"] + sorted(DEFECT_CLASSES)  # 9 classes

# ------------------------------------------------------------
# Run inference on all test images
# ------------------------------------------------------------
def collect_predictions(model):
    y_true, y_pred, records = [], [], []
    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        true_label = class_dir.name
        for img_file in sorted(class_dir.glob("*.*")):
            if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}:
                continue
            pred_label = predict_image(model, img_file)
            y_true.append(true_label)
            y_pred.append(pred_label)
            records.append({
                "file": str(img_file),
                "true": true_label,
                "pred": pred_label,
                "correct": true_label == pred_label,
            })
    return y_true, y_pred, records

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

def plot_per_class_metrics(report_dict, labels, out_path):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(metrics):
        vals = [report_dict.get(l, {}).get(m, 0) for l in labels]
        bars = ax.bar(x + i * width, vals, width, label=m.capitalize())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1 — Test Set")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

def plot_per_class_accuracy(y_true, y_pred, labels, out_path):
    correct = defaultdict(int)
    total = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1

    accs = [correct[l] / total[l] if total[l] > 0 else 0 for l in labels]
    counts = [total[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accs, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
    for bar, a, c in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{a:.1%}\n(n={c})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy — Test Set")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model …")
    model = MultiTaskResNet50(num_defect_classes=len(DEFECT_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Predict
    print("Running inference on test images …")
    y_true, y_pred, records = collect_predictions(model)
    print(f"  Total images: {len(y_true)}")

    # Present labels (only labels that appear in data)
    present_labels = sorted(set(y_true) | set(y_pred))
    # keep ALL_CLASSES order but only those present
    labels = [l for l in ALL_CLASSES if l in present_labels]

    # Metrics
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n  Overall Accuracy: {overall_acc:.4f} ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})")

    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    print(f"\n{report_str}")

    # Per-class accuracy
    per_class = {}
    for l in labels:
        total_l = sum(1 for t in y_true if t == l)
        correct_l = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        per_class[l] = {
            "accuracy": correct_l / total_l if total_l else 0,
            "correct": correct_l,
            "total": total_l,
            "precision": report_dict[l]["precision"],
            "recall": report_dict[l]["recall"],
            "f1": report_dict[l]["f1-score"],
            "support": report_dict[l]["support"],
        }

    # Misclassified images
    misclassified = [r for r in records if not r["correct"]]

    # Save JSON summary
    summary = {
        "total_images": len(y_true),
        "overall_accuracy": round(overall_acc, 4),
        "macro_avg": {
            "precision": round(report_dict["macro avg"]["precision"], 4),
            "recall": round(report_dict["macro avg"]["recall"], 4),
            "f1": round(report_dict["macro avg"]["f1-score"], 4),
        },
        "weighted_avg": {
            "precision": round(report_dict["weighted avg"]["precision"], 4),
            "recall": round(report_dict["weighted avg"]["recall"], 4),
            "f1": round(report_dict["weighted avg"]["f1-score"], 4),
        },
        "per_class": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items()} for k, v in per_class.items()},
        "num_misclassified": len(misclassified),
        "misclassified_samples": misclassified[:50],  # first 50
    }

    json_path = OUT_DIR / "test_evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {json_path}")

    # Save classification report text
    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
        f.write(report_str)
    print(f"  Saved {OUT_DIR / 'classification_report.txt'}")

    # Plots
    print("\nGenerating plots …")
    plot_confusion_matrix(y_true, y_pred, labels, OUT_DIR / "confusion_matrix.png")
    plot_per_class_metrics(report_dict, labels, OUT_DIR / "per_class_metrics.png")
    plot_per_class_accuracy(y_true, y_pred, labels, OUT_DIR / "per_class_accuracy.png")

    print("\nDone! All results in:", OUT_DIR)

if __name__ == "__main__":
    main()
