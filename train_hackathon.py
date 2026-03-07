import os, time, json
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

import torchvision.transforms as T
import torchvision.models as models

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = "Data"        # <-- change if your folder name is different
OUT_DIR = "reports"
SEED = 42
IMG_SIZE = 224            # ResNet standard
BATCH_SIZE = 32
EPOCHS = 12               # small dataset => 10-20 is fine
LR = 2e-4                 # transfer learning LR
NUM_WORKERS = 0           # Windows: keep 0 for stability

DEVICE = "cpu"            # you are CPU-only

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Dataset
# -----------------------------
class WaferDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            for f in os.listdir(cls_path):
                if f.lower().endswith((".png",".jpg",".jpeg",".tif",".bmp")):
                    self.samples.append((os.path.join(cls_path, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("L")     # grayscale
        img = img.convert("RGB")               # replicate channel to RGB for pretrained model
        if self.transform:
            img = self.transform(img)
        return img, y


transform_train = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
])

transform_eval = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

dataset_full = WaferDataset(DATA_ROOT, transform=None)
num_classes = len(dataset_full.classes)

# Split
n = len(dataset_full)
train_len = int(0.7*n)
val_len = int(0.15*n)
test_len = n - train_len - val_len

# Use separate datasets for transform differences
dataset_train = WaferDataset(DATA_ROOT, transform=transform_train)
dataset_eval  = WaferDataset(DATA_ROOT, transform=transform_eval)

train_ds, val_ds, test_ds = random_split(
    dataset_train, [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(SEED)
)
# For val/test use eval transforms:
val_ds.dataset = dataset_eval
test_ds.dataset = dataset_eval

# -----------------------------
# Class imbalance handling
# -----------------------------
train_targets = [dataset_full.samples[i][1] for i in train_ds.indices]
counts = Counter(train_targets)

class_weights = {c: 1.0/max(counts[c], 1) for c in range(num_classes)}
sample_weights = [class_weights[t] for t in train_targets]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Save class counts
with open(os.path.join(OUT_DIR, "train_class_counts.json"), "w") as f:
    json.dump({dataset_full.classes[k]: int(v) for k,v in counts.items()}, f, indent=2)

# -----------------------------
# Model (Transfer Learning)
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Train
# -----------------------------
best_val = 0.0
best_path = os.path.join(OUT_DIR, "best_model.pt")

for epoch in range(1, EPOCHS+1):
    model.train()
    tr_correct = 0
    tr_total = 0
    tr_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * y.size(0)
        tr_correct += (logits.argmax(1) == y).sum().item()
        tr_total += y.size(0)

    tr_acc = tr_correct / tr_total
    tr_loss = tr_loss / tr_total

    # Validation
    model.eval()
    va_correct = 0
    va_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            va_correct += (logits.argmax(1) == y).sum().item()
            va_total += y.size(0)

    va_acc = va_correct / va_total if va_total else 0.0

    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss {tr_loss:.4f} | Train Acc {tr_acc:.3f} | Val Acc {va_acc:.3f}")

    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), best_path)

print("Best Val Acc:", best_val)

# -----------------------------
# Test + Metrics
# -----------------------------
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval()

all_preds = []
all_true = []

# Inference timing
t0 = time.time()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_true.extend(y.numpy().tolist())
t1 = time.time()

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

test_acc = (all_preds == all_true).mean()
avg_time_per_img = (t1 - t0) / len(test_ds)

print("\nTEST ACC:", test_acc)
print("AVG INFERENCE TIME PER IMAGE (s):", avg_time_per_img)

# Confusion matrix
cm = confusion_matrix(all_true, all_preds, labels=list(range(num_classes)))
np.savetxt(os.path.join(OUT_DIR, "confusion_matrix.csv"), cm, delimiter=",")

# Classification report
report = classification_report(all_true, all_preds, target_names=dataset_full.classes, digits=4)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
print("\n", report)

# -----------------------------
# Required plot: Accuracy vs Occurrence
# -----------------------------
# Occurrence from train distribution; per-class accuracy from test confusion matrix
occurrence = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=float)
per_class_acc = (cm.diagonal() / np.maximum(cm.sum(axis=1), 1)).astype(float)

plt.figure()
plt.scatter(occurrence, per_class_acc)
for i, name in enumerate(dataset_full.classes):
    plt.annotate(name, (occurrence[i], per_class_acc[i]))

plt.xlabel("Class Occurrence in Train Set")
plt.ylabel("Per-Class Accuracy on Test Set")
plt.title("Accuracy vs Class Occurrence")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "accuracy_vs_occurrence.png"), dpi=200)
plt.show()

# Save numeric values for plotting in slides
np.savetxt(os.path.join(OUT_DIR, "acc_vs_occurrence.csv"),
           np.stack([occurrence, per_class_acc], axis=1),
           delimiter=",",
           header="occurrence,per_class_accuracy",
           comments="")