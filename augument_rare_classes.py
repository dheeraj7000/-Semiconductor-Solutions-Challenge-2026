"""
Data Augmentation for rare defect classes (defect3, defect4).

Two complementary strategies, unified in one script:

A) Patch Amplification (defect3 + defect4):
   - Load each original defect image
   - Apply localized augmentations: rotation, scale, brightness, contrast,
     elastic distortion, blur, noise, flips
   - Save synthetic samples to reach TARGET_PER_CLASS

B) Procedural Speckle Generation (defect4 only):
   - Load "good" wafer images as base templates
   - Inject synthetic particle contamination (Gaussian blobs, streaks)
   - Produces realistic defect4 patterns from clean backgrounds

Target: bring defect3 (9 → ~250) and defect4 (14 → ~250+)
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import gaussian_filter, map_coordinates
from pathlib import Path


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT = Path("Data")
GOOD_DIR = DATA_ROOT / "good"
TARGET_PER_CLASS = 250  # target total samples per class (originals + synthetic)
NUM_SPECKLE_IMAGES = 50  # additional speckle-generated defect4 images

RARE_CLASSES = ["defect3", "defect4"]
VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}


# ===========================================================================
# Augmentation primitives (Patch Amplification)
# ===========================================================================

def random_rotation(img, max_angle=10):
    """Small rotation ±max_angle degrees."""
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)


def random_scale(img, scale_range=(0.9, 1.1)):
    """Slight scaling, then center-crop/pad back to original size."""
    w, h = img.size
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = img.resize((new_w, new_h), Image.BICUBIC)

    result = Image.new(img.mode, (w, h), 0)
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2

    if scale >= 1.0:
        crop_x = (new_w - w) // 2
        crop_y = (new_h - h) // 2
        result = scaled.crop((crop_x, crop_y, crop_x + w, crop_y + h))
    else:
        result.paste(scaled, (paste_x, paste_y))

    return result


def random_brightness(img, factor_range=(0.85, 1.15)):
    """Mild brightness shift."""
    factor = random.uniform(*factor_range)
    return ImageEnhance.Brightness(img).enhance(factor)


def random_contrast(img, factor_range=(0.9, 1.1)):
    """Mild contrast shift."""
    factor = random.uniform(*factor_range)
    return ImageEnhance.Contrast(img).enhance(factor)


def elastic_distortion(img, alpha=15, sigma=3):
    """Light elastic distortion using scipy."""
    arr = np.array(img, dtype=np.float32)
    is_color = arr.ndim == 3

    shape = arr.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices_y = np.clip(y + dy, 0, shape[0] - 1)
    indices_x = np.clip(x + dx, 0, shape[1] - 1)

    if is_color:
        channels = []
        for c in range(arr.shape[2]):
            warped = map_coordinates(arr[:, :, c], [indices_y, indices_x],
                                     order=1, mode='reflect')
            channels.append(warped)
        result = np.stack(channels, axis=-1)
    else:
        result = map_coordinates(arr, [indices_y, indices_x],
                                 order=1, mode='reflect')

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def gaussian_blur(img, radius_range=(0.3, 1.0)):
    """Light Gaussian blur."""
    radius = random.uniform(*radius_range)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_gaussian_noise(img, std_range=(3, 10)):
    """Add Gaussian noise."""
    arr = np.array(img, dtype=np.float32)
    std = random.uniform(*std_range)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_salt_pepper_noise(img, amount=0.005):
    """Add salt-and-pepper noise."""
    arr = np.array(img)
    num_salt = int(amount * arr.size)
    coords = tuple(np.random.randint(0, max(1, d), num_salt) for d in arr.shape)
    arr[coords] = 255
    num_pepper = int(amount * arr.size)
    coords = tuple(np.random.randint(0, max(1, d), num_pepper) for d in arr.shape)
    arr[coords] = 0
    return Image.fromarray(arr)


def random_flip(img):
    """Random horizontal and/or vertical flip."""
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


# ===========================================================================
# Patch amplification pipeline
# ===========================================================================

def augment_image(img):
    """Apply a random combination of augmentations to a single image."""
    img = random_flip(img)

    if random.random() < 0.8:
        img = random_rotation(img, max_angle=10)
    if random.random() < 0.5:
        img = random_scale(img, scale_range=(0.9, 1.1))
    if random.random() < 0.7:
        img = random_brightness(img, factor_range=(0.85, 1.15))
    if random.random() < 0.5:
        img = random_contrast(img, factor_range=(0.9, 1.1))
    if random.random() < 0.4:
        img = elastic_distortion(img, alpha=12, sigma=3)
    if random.random() < 0.4:
        img = gaussian_blur(img, radius_range=(0.3, 0.8))

    noise_roll = random.random()
    if noise_roll < 0.3:
        img = add_gaussian_noise(img, std_range=(3, 8))
    elif noise_roll < 0.45:
        img = add_salt_pepper_noise(img, amount=0.003)

    return img


def load_originals(class_dir):
    """Load all original images from a class directory."""
    images = []
    for f in sorted(class_dir.iterdir()):
        if f.suffix.lower() in VALID_EXT:
            img = Image.open(f)
            images.append((f.name, img.copy()))
    return images


def generate_patch_augmented(class_name, originals, target_count):
    """Generate synthetic augmented images via patch amplification."""
    current_count = len(originals)
    needed = target_count - current_count

    if needed <= 0:
        print(f"  {class_name}: already has {current_count} >= {target_count}, skipping patch amplification.")
        return []

    augments_per_image = needed // current_count
    remainder = needed % current_count

    print(f"  {class_name} [patch]: {current_count} originals -> generating {needed} synthetic")
    print(f"    ~{augments_per_image} augmentations per image (+{remainder} extra)")

    synthetic = []
    gen_idx = 0
    for img_idx, (orig_name, orig_img) in enumerate(originals):
        n_aug = augments_per_image + (1 if img_idx < remainder else 0)

        for _ in range(n_aug):
            aug_img = augment_image(orig_img.copy())
            stem = Path(orig_name).stem
            new_name = f"syn_{class_name}_{gen_idx:04d}_from_{stem[:16]}.PNG"
            synthetic.append((new_name, aug_img))
            gen_idx += 1

    return synthetic


# ===========================================================================
# Procedural speckle generation (defect4 only)
# ===========================================================================

def create_gaussian_spot(radius):
    """Create a 2D Gaussian spot kernel."""
    size = radius * 4 + 1
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    sigma = radius / 2.0
    return np.exp(-dist_sq / (2 * sigma ** 2))


def add_light_streak(arr):
    """Add a subtle streak artifact (rare feature in defect4)."""
    h, w = arr.shape[:2]
    is_color = arr.ndim == 3

    y_start = random.randint(0, h - 1)
    x_start = random.randint(0, w - 1)
    length = random.randint(20, min(80, w // 3))
    angle = random.uniform(-15, 15)

    intensity = random.uniform(8, 20)
    thickness = random.randint(1, 2)

    for i in range(length):
        x = int(x_start + i * np.cos(np.radians(angle)))
        y = int(y_start + i * np.sin(np.radians(angle)))

        for dy in range(-thickness, thickness + 1):
            yy = y + dy
            if 0 <= yy < h and 0 <= x < w:
                fade = 1.0 - abs(dy) / (thickness + 1)
                if is_color:
                    arr[yy, x, :] += intensity * fade
                else:
                    arr[yy, x] += intensity * fade

    return arr


def inject_speckles(img_array, num_spots=None):
    """
    Inject synthetic particle defects into a grayscale/RGB image array.
    Mimics defect4 pattern: scattered small bright/dark spots on wafer surface.
    """
    arr = img_array.astype(np.float32)
    h, w = arr.shape[:2]
    is_color = arr.ndim == 3

    if num_spots is None:
        num_spots = random.randint(15, 50)

    for _ in range(num_spots):
        margin_y = int(h * 0.05)
        margin_x = int(w * 0.05)
        cy = random.randint(margin_y, h - margin_y - 1)
        cx = random.randint(margin_x, w - margin_x - 1)

        radius = random.randint(2, 6)

        if random.random() < 0.75:
            intensity = random.uniform(15, 45)
        else:
            intensity = random.uniform(-35, -10)

        spot = create_gaussian_spot(radius)
        spot_h, spot_w = spot.shape

        y1 = cy - spot_h // 2
        y2 = y1 + spot_h
        x1 = cx - spot_w // 2
        x2 = x1 + spot_w

        sy1 = max(0, -y1)
        sy2 = spot_h - max(0, y2 - h)
        sx1 = max(0, -x1)
        sx2 = spot_w - max(0, x2 - w)

        y1 = max(0, y1)
        y2 = min(h, y2)
        x1 = max(0, x1)
        x2 = min(w, x2)

        if y2 <= y1 or x2 <= x1:
            continue

        spot_crop = spot[sy1:sy2, sx1:sx2]

        if is_color:
            for c in range(arr.shape[2]):
                arr[y1:y2, x1:x2, c] += spot_crop * intensity
        else:
            arr[y1:y2, x1:x2] += spot_crop * intensity

    if random.random() < 0.15:
        arr = add_light_streak(arr)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_good_images(max_images=200):
    """Load a random subset of good images as base templates."""
    all_files = sorted([
        f for f in GOOD_DIR.iterdir()
        if f.suffix.lower() in VALID_EXT
    ])
    if len(all_files) > max_images:
        all_files = random.sample(all_files, max_images)
    return all_files


def generate_speckle_images(good_files, num_to_generate):
    """Generate synthetic defect4 images from good base images."""
    synthetic = []

    for i in range(num_to_generate):
        base_path = random.choice(good_files)
        base_img = Image.open(base_path).convert("RGB")
        arr = np.array(base_img)

        num_spots = random.randint(15, 50)
        arr = inject_speckles(arr, num_spots=num_spots)

        result = Image.fromarray(arr)

        if random.random() < 0.3:
            result = result.filter(ImageFilter.GaussianBlur(radius=0.4))
        if random.random() < 0.4:
            factor = random.uniform(0.93, 1.07)
            result = ImageEnhance.Brightness(result).enhance(factor)

        name = f"speckle_defect4_{i:04d}.PNG"
        synthetic.append((name, result))

    return synthetic


# ===========================================================================
# Main — unified pipeline
# ===========================================================================

def save_images(image_list, output_dir):
    """Save a list of (name, PIL.Image) to output_dir. Returns count saved."""
    saved = 0
    for name, img in image_list:
        out_path = output_dir / name
        img.save(str(out_path))
        saved += 1
    return saved


def main():
    print("=" * 60)
    print("RARE CLASS AUGMENTATION — Patch Amplification + Speckle Gen")
    print("=" * 60)

    total_saved = {}

    # ----- Stage 1: Patch amplification for all rare classes -----
    print("\n--- Stage 1: Patch Amplification (defect3 + defect4) ---")
    for class_name in RARE_CLASSES:
        class_dir = DATA_ROOT / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found, skipping.")
            continue

        print(f"\nProcessing: {class_name}")
        originals = load_originals(class_dir)
        print(f"  Loaded {len(originals)} original images")

        synthetic = generate_patch_augmented(class_name, originals, TARGET_PER_CLASS)
        saved = save_images(synthetic, class_dir)
        total_saved[class_name] = total_saved.get(class_name, 0) + saved
        print(f"  Saved {saved} patch-augmented images")

    # ----- Stage 2: Speckle generation for defect4 -----
    print(f"\n--- Stage 2: Speckle Generation (defect4 only) ---")
    defect4_dir = DATA_ROOT / "defect4"
    if defect4_dir.exists() and GOOD_DIR.exists():
        good_files = load_good_images(max_images=200)
        print(f"  Loaded {len(good_files)} good images as base templates")
        print(f"  Generating {NUM_SPECKLE_IMAGES} speckle-based defect4 images")

        speckle_synthetic = generate_speckle_images(good_files, NUM_SPECKLE_IMAGES)
        saved = save_images(speckle_synthetic, defect4_dir)
        total_saved["defect4"] = total_saved.get("defect4", 0) + saved
        print(f"  Saved {saved} speckle-generated images")
    else:
        if not defect4_dir.exists():
            print(f"  WARNING: {defect4_dir} not found, skipping speckle generation.")
        if not GOOD_DIR.exists():
            print(f"  WARNING: {GOOD_DIR} not found, cannot generate speckle images.")

    # ----- Final summary -----
    print(f"\n{'=' * 60}")
    print("FINAL CLASS COUNTS:")
    print(f"{'=' * 60}")
    for class_name in sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir()):
        class_dir = DATA_ROOT / class_name
        count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXT)
        marker = ""
        if class_name == "defect4":
            marker = " <- patch + speckle augmented"
        elif class_name == "defect3":
            marker = " <- patch augmented"
        print(f"  {class_name:>10s}: {count:>5d}{marker}")

    print(f"\nSynthetic images generated: {total_saved}")


if __name__ == "__main__":
    main()
