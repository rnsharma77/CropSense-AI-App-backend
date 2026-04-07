"""
CropSense AI - ML Training Pipeline
===================================
Loads labeled images from either:
1. your /api/dataset endpoint, or
2. MongoDB directly for local CRA development,
3. a local dataset JSON export,
then trains MobileNetV3-Small and saves the model to server/ml/models/

Usage:
    cd server
    python3 ml/train.py
    python3 ml/train.py --epochs 30 --min-samples 20
    python3 ml/train.py --data-source mongo --mongo-uri <mongodb-uri>
    python3 ml/train.py --data-source file --dataset-file ml/models/dataset_export.json
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

from activity_log import append_log, ensure_log_file

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

API_BASE = os.getenv("API_BASE", "http://localhost:3000")
DEFAULT_MONGO_URI = os.getenv("MONGODB_URI")

# By default models and indexes are saved under server/ml_model/models
ML_MODEL_DIR = Path(__file__).parent.parent / "ml_model" / "models"
MODEL_DIR = ML_MODEL_DIR
MODEL_PATH = MODEL_DIR / "cropsense_model.pth"
INDEX_PATH = MODEL_DIR / "class_index.json"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ensure_log_file()
print(f"Device: {DEVICE}")
print(f"API:    {API_BASE}")


class CropDiseaseDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(io.BytesIO(base64.b64decode(record["imageBase64"]))).convert("RGB")
        label = self.class_to_idx[record["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


train_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def fetch_records_from_api(min_confidence):
    response = requests.get(
        f"{API_BASE}/api/dataset",
        params={
            "action": "export",
            "onlyVerified": "true",
            "minConfidence": min_confidence,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("dataset", [])


def fetch_records_from_mongo(mongo_uri, min_confidence):
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed. Run: pip install -r server/ml/requirements.txt")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI is not set. Pass --mongo-uri or configure the env var.")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=15000,
    )
    try:
        collection = client.get_default_database().collection("analyses")
        cursor = collection.find(
            {
                "imageBase64": {"$ne": None},
                "isDemo": {"$ne": True},
                "isVerified": True,
                "confidence": {"$gte": min_confidence},
            },
            {
                "verifiedLabel": 1,
                "disease": 1,
                "confidence": 1,
                "severityScore": 1,
                "imageBase64": 1,
                "timestamp": 1,
            },
        )

        records = []
        for row in cursor:
            label = row.get("verifiedLabel") or row.get("disease")
            if not label or not row.get("imageBase64"):
                continue
            records.append(
                {
                    "id": str(row.get("_id")),
                    "label": label,
                    "confidence": row.get("confidence"),
                    "severityScore": row.get("severityScore", 0),
                    "imageBase64": row.get("imageBase64"),
                    "createdAt": row.get("timestamp"),
                }
            )
        return records
    finally:
        client.close()


def fetch_records_from_file(dataset_file):
    if not dataset_file:
        raise RuntimeError("Pass --dataset-file when using --data-source file.")
    dataset_path = Path(dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        return data.get("dataset", [])
    if isinstance(data, list):
        return data
    raise RuntimeError("Dataset file must contain a list or an object with a dataset field.")


def fetch_records_from_folder(dataset_dir, min_confidence=0.0):
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset folder not found: {dataset_path}")

    records = []
    # Expect subfolders per class name
    for class_dir in sorted([p for p in dataset_path.iterdir() if p.is_dir()]):
        label = class_dir.name
        for img_path in class_dir.glob('*'):
            if not img_path.is_file():
                continue
            try:
                with open(img_path, 'rb') as fh:
                    b64 = base64.b64encode(fh.read()).decode('utf-8')
                records.append({
                    'id': str(img_path),
                    'label': label,
                    'confidence': 1.0,
                    'severityScore': 0,
                    'imageBase64': b64,
                    'createdAt': None,
                })
            except Exception:
                continue
    return records


def load_records(args):
    errors = []
    if args.data_source == "auto":
        sources = ["api", "mongo"]
    else:
        sources = [args.data_source]

    for source in sources:
        try:
            if source == "api":
                print(f"\n[1/5] Fetching dataset from {API_BASE}/api/dataset ...")
                records = fetch_records_from_api(args.min_confidence)
            elif source == "mongo":
                print("\n[1/5] Fetching dataset directly from MongoDB ...")
                records = fetch_records_from_mongo(args.mongo_uri or DEFAULT_MONGO_URI, args.min_confidence)
            elif source == 'folder':
                print(f"\n[1/5] Loading dataset from folder: {args.dataset_dir} ...")
                records = fetch_records_from_folder(args.dataset_dir, args.min_confidence)
            else:
                print("\n[1/5] Loading dataset from local JSON file ...")
                records = fetch_records_from_file(args.dataset_file)

            print(f"  Records: {len(records)}")
            if records:
                print(f"  Source:  {source}")
            return records, source
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            print(f"  {source} failed: {exc}")

    print("\nUnable to load a training dataset.")
    for error in errors:
        print(f"  - {error}")
    print("If you are running CRA locally, use --data-source file or --data-source mongo.")
    sys.exit(1)


def build_splits(records, class_to_idx, val_ratio, seed):
    labels = [class_to_idx[record["label"]] for record in records]
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        print("Training requires at least 2 classes after filtering.")
        sys.exit(1)

    class_counts = Counter(labels)
    if min(class_counts.values()) < 2:
        print("Each trainable class needs at least 2 samples so one can reach validation.")
        sys.exit(1)

    indices = list(range(len(records)))
    val_count = max(len(unique_labels), int(round(len(records) * val_ratio)))
    if val_count >= len(records):
        val_count = len(records) - 1

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_count,
        stratify=labels,
        random_state=seed,
    )

    train_ds = Subset(CropDiseaseDataset(records, class_to_idx, train_tf), train_idx)
    val_ds = Subset(CropDiseaseDataset(records, class_to_idx, val_tf), val_idx)
    return train_ds, val_ds


def build_model(num_classes, use_pretrained):
    if use_pretrained:
        try:
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
        except Exception as exc:
            print(f"  Pretrained weights unavailable ({exc}). Falling back to random initialization.")
            model = models.mobilenet_v3_small(weights=None)
    else:
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def train_one(model, loader, criterion, optimizer):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += images.size(0)

    return loss_sum / total, 100 * correct / total


def eval_one(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    preds = []
    gt = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)

            predicted = outputs.argmax(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
            preds.extend(predicted.cpu().numpy())
            gt.extend(labels.cpu().numpy())

    return loss_sum / total, 100 * correct / total, preds, gt


def main(args):
    records, source = load_records(args)

    if not records:
        print("No labeled data. Save analyses with imageBase64 and click 'Verify' in the Analyses page.")
        sys.exit(1)

    counts = Counter(record["label"] for record in records)
    valid_classes = sorted(name for name, count in counts.items() if count >= args.min_samples)
    print(f"\n[2/5] Classes with >= {args.min_samples} samples: {valid_classes}")

    if not valid_classes:
        print(f"No class has {args.min_samples}+ samples. Counts: {dict(counts)}")
        sys.exit(1)

    valid_records = [record for record in records if record["label"] in valid_classes]
    print(f"  Valid records: {len(valid_records)}")
    for record in valid_records:
        append_log(
            event_type="TRAIN",
            disease_name=record["label"],
            record_id=record.get("id", ""),
            source=source,
            confidence=record.get("confidence", ""),
            notes="Included in training dataset",
        )

    class_to_idx = {name: index for index, name in enumerate(valid_classes)}
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"  Class map: {class_to_idx}")

    print("\n[3/5] Preparing dataset...")
    train_ds, val_ds = build_splits(
        valid_records,
        class_to_idx,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    print(f"\n[4/5] Training MobileNetV3-Small - {args.epochs} epochs...")
    model = build_model(num_classes, use_pretrained=not args.no_pretrained)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0.0
    last_preds = []
    last_gt = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_acc = train_one(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_gt = eval_one(model, val_loader, criterion)
        scheduler.step()
        last_preds, last_gt = val_preds, val_gt

        print(
            f"  Epoch {epoch:3d}/{args.epochs}  "
            f"train {train_acc:.1f}%  val {val_acc:.1f}%  ({time.time() - started:.1f}s)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "num_classes": num_classes,
                    "val_acc": val_acc,
                    "img_size": IMG_SIZE,
                    "data_source": source,
                },
                MODEL_PATH,
            )
            print(f"  Best model saved (val={val_acc:.1f}%)")

    print("\n[5/5] Saving class index...")
    with open(INDEX_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(key): value for key, value in idx_to_class.items()},
                "num_classes": num_classes,
                "classes": valid_classes,
                "img_size": IMG_SIZE,
                "best_val_acc": best_acc,
                "data_source": source,
            },
            handle,
            indent=2,
        )

    labels = [idx_to_class[index] for index in sorted(idx_to_class)]
    print(f"\n{'=' * 50}")
    print(f"Done! Best val accuracy: {best_acc:.2f}%")
    print(f"Source: {source}")
    print(f"Model -> {MODEL_PATH}")
    print(f"Index -> {INDEX_PATH}")
    print(f"{'=' * 50}")
    print(classification_report(last_gt, last_preds, target_names=labels, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mongo-uri", type=str)
    parser.add_argument("--dataset-file", type=str)
    parser.add_argument("--dataset-dir", type=str, default=str(Path(__file__).parent.parent / "ml_model" / "dataset"))
    parser.add_argument("--data-source", choices=["auto", "api", "mongo", "file", "folder"], default="auto")
    parser.add_argument("--output-dir", type=str, default=str(ML_MODEL_DIR))
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

        # if output-dir provided, override MODEL_PATHs
    if args.output_dir:
        MODEL_DIR = Path(args.output_dir)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH = MODEL_DIR / "cropsense_model.pth"
        INDEX_PATH = MODEL_DIR / "class_index.json"

        main(args)
