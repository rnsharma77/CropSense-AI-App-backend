"""
CropSense AI — ML Prediction (Local Model Inference)
======================================================
Once your model is trained, this script predicts disease
from a base64 image without calling Plant.id API.

Usage:
    python3 server/ml/predict.py --file path/to/leaf.jpg
    python3 server/ml/predict.py --image <base64_string>

Output: JSON printed to stdout — parsed by Node.js
"""

import sys, json, base64, io, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from activity_log import append_log, ensure_log_file

# Prefer models stored under ../ml_model/models (project-level ML artifacts).
# Fallback to the original ml/models directory if not present.
ALT_MODEL_DIR = Path(__file__).parent.parent / 'ml_model' / 'models'
DEFAULT_MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR = ALT_MODEL_DIR if ALT_MODEL_DIR.exists() else DEFAULT_MODEL_DIR
MODEL_PATH = MODEL_DIR / 'cropsense_model.pth'
DEVICE     = torch.device('cpu')   # CPU for production inference

ensure_log_file()


def sanitize_base64(value):
    return str(value or '').strip().split(',', 1)[-1]


def resolve_idx_to_class(raw_idx_to_class):
    if not raw_idx_to_class:
        return {}

    resolved = {}
    for key, value in raw_idx_to_class.items():
        try:
            resolved[int(key)] = value
        except (TypeError, ValueError):
            continue
    return resolved


def load():
    if not MODEL_PATH.exists():
        return None, None
    ck  = torch.load(MODEL_PATH, map_location=DEVICE)
    m   = models.mobilenet_v3_small(weights=None)
    inf = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(inf, ck['num_classes'])
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m, ck


def predict(model, ck, img_bytes):
    sz = ck.get('img_size', 224)
    tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img    = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tensor = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1)[0]

    idx_to_class = resolve_idx_to_class(ck.get('idx_to_class'))
    k    = min(3, len(idx_to_class))
    vals, idxs = torch.topk(probs, k)
    preds = [
        {
            'disease': idx_to_class.get(i.item(), f'class_{i.item()}'),
            'probability': round(p.item(), 4),
        }
        for p, i in zip(vals, idxs)
    ]
    top   = preds[0]

    return {
        'success':        True,
        'source':         'local_ml',
        'disease':        top['disease'],
        'confidence':     top['probability'],
        'severityScore':  int(top['probability'] * 100),
        'severity':       'High' if top['probability']>0.7 else 'Medium' if top['probability']>0.4 else 'Low',
        'topPredictions': preds,
        'modelAccuracy':  ck.get('val_acc', 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str)
    ap.add_argument('--file',  type=str)
    args = ap.parse_args()

    model, ck = load()
    if model is None:
        print(json.dumps({'success':False,'source':'local_ml','error':'Model not trained yet. Run: python3 server/ml/train.py'}))
        sys.exit(0)   # Exit 0 so Node.js falls back to Plant.id gracefully

    if args.image:
        img_bytes = base64.b64decode(sanitize_base64(args.image))
    elif args.file:
        img_bytes = open(args.file,'rb').read()
    else:
        print(json.dumps({'success':False,'error':'No image provided'}))
        sys.exit(1)

    result = predict(model, ck, img_bytes)
    append_log(
        event_type='DETECTED',
        disease_name=result['disease'],
        source=result.get('source', 'local_ml'),
        confidence=result.get('confidence', ''),
        notes='Predicted by local ML model',
    )
    print(json.dumps(result))


if __name__ == '__main__':
    main()
