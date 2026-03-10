"""
Defect Classification Inference Application
============================================
Standalone application for detecting and classifying defects from grayscale images.
Handles images up to 1500x2500 pixels. Target: <1 second per image.

Usage:
    # Single image
    python inference.py --image path/to/image.png --model_path reports/best_model.pt

    # Batch directory
    python inference.py --image_dir path/to/images/ --model_path reports/best_model.pt

    # Benchmark mode
    python inference.py --benchmark --model_path reports/best_model.pt
"""

import os, sys, time, json, argparse
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
import timm

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CLASSES = ["defect1", "defect10", "defect2", "defect3", "defect4",
                   "defect5", "defect8", "defect9", "good"]


def load_model(model_path, model_name="convnext_base", num_classes=9, device="cpu"):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def preprocess(image_path, img_size=384):
    """Load a grayscale image and prepare for inference."""
    img = Image.open(image_path).convert("L")
    img_rgb = Image.merge("RGB", [img, img, img])
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(img_rgb).unsqueeze(0), img.size  # (1, 3, H, W), original_size


@torch.no_grad()
def predict(model, image_tensor, device, class_names=None):
    """Run inference and return prediction with confidence."""
    if class_names is None:
        class_names = DEFAULT_CLASSES

    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs = F.softmax(logits, dim=1)
    conf, pred_idx = probs.max(1)

    pred_class = class_names[pred_idx.item()]
    confidence = conf.item()

    # Top-3 predictions
    topk = torch.topk(probs, min(3, len(class_names)), dim=1)
    top3 = [(class_names[i], p) for i, p in zip(topk.indices[0].tolist(), topk.values[0].tolist())]

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top3": top3,
        "all_probs": {class_names[i]: probs[0, i].item() for i in range(len(class_names))}
    }


def classify_image(image_path, model, device, img_size=260, class_names=None):
    """Full pipeline: load, preprocess, predict, time it."""
    t0 = time.perf_counter()
    tensor, original_size = preprocess(image_path, img_size)
    result = predict(model, tensor, device, class_names)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    result["inference_time_ms"] = (t1 - t0) * 1000
    result["original_size"] = original_size
    result["image_path"] = str(image_path)
    return result


def benchmark(model, device, img_size=260, n_runs=100):
    """Benchmark inference speed at various resolutions."""
    print("\nInference Benchmark")
    print("=" * 50)

    resolutions = [
        ("Training res", (img_size, img_size)),
        ("Medium", (800, 1000)),
        ("Large", (1500, 2500)),
    ]

    results = {}
    for name, (h, w) in resolutions:
        dummy = torch.randn(1, 3, h, w).to(device)

        # Warmup
        for _ in range(10):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg = np.mean(times)
        std = np.std(times)
        p95 = np.percentile(times, 95)
        results[name] = {"resolution": f"{h}x{w}", "avg_ms": avg, "std_ms": std, "p95_ms": p95}
        status = "PASS" if avg < 1000 else "FAIL"
        print(f"  {name:15s} ({h:>4d}x{w:<4d}): {avg:7.1f} ± {std:.1f} ms (p95: {p95:.1f} ms) [{status}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Defect Classification Inference")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--model_path", default="reports/best_model.pt")
    parser.add_argument("--model_name", default="convnext_base")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model_path, args.model_name, args.num_classes, device)
    print(f"Model loaded: {args.model_path}")

    if args.benchmark:
        results = benchmark(model, device, args.img_size)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
        return

    all_results = []

    if args.image:
        result = classify_image(args.image, model, device, args.img_size)
        print(f"\nImage: {result['image_path']}")
        print(f"  Prediction: {result['predicted_class']} ({result['confidence']:.1%})")
        print(f"  Top-3: {result['top3']}")
        print(f"  Time: {result['inference_time_ms']:.1f} ms")
        all_results.append(result)

    elif args.image_dir:
        img_dir = Path(args.image_dir)
        valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
        images = sorted([f for f in img_dir.rglob("*") if f.suffix.lower() in valid_ext])
        print(f"\nProcessing {len(images)} images from {img_dir}")

        for img_path in images:
            result = classify_image(img_path, model, device, args.img_size)
            print(f"  {img_path.name}: {result['predicted_class']} "
                  f"({result['confidence']:.1%}, {result['inference_time_ms']:.0f}ms)")
            all_results.append(result)

        # Summary
        times = [r["inference_time_ms"] for r in all_results]
        print(f"\nProcessed {len(all_results)} images")
        print(f"  Avg time: {np.mean(times):.1f} ms")
        print(f"  Max time: {np.max(times):.1f} ms")

    if args.output and all_results:
        # Clean up for JSON serialization
        for r in all_results:
            r["top3"] = [(c, float(p)) for c, p in r["top3"]]
            r["all_probs"] = {k: float(v) for k, v in r["all_probs"].items()}
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
