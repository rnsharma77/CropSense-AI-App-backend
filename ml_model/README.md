ML model folder
================

Place your training dataset here before running training.

Dataset layout
--------------
- Create a folder per class under `server/ml_model/dataset`, e.g.:

  server/ml_model/dataset/
  ├─ Late_blight/
  │  ├─ img01.jpg
  │  └─ img02.jpg
  ├─ Healthy/
  │  ├─ img01.jpg
  │  └─ img02.jpg
  └─ Bacterial_spot/
     ├─ img01.jpg
     └─ img02.jpg

- Supported image formats: `.jpg`, `.jpeg`, `.png`.
- Filenames are arbitrary; class is derived from the folder name.

Training
--------
From the project root run (recommended inside a Python virtualenv):

```bash
cd my-app/server
pip install -r ml/requirements.txt
python ml/train.py --data-source folder --dataset-dir ../ml_model/dataset --output-dir ../ml_model/models --epochs 20 --batch-size 16
```

This will save `cropsense_model.pth` and `class_index.json` under `server/ml_model/models`.

Prediction (server)
-------------------
The backend exposes a local endpoint `/api/local_predict` that accepts a JSON body `{ "imageBase64": "..." }` and returns the same structure used by the UI. The frontend will automatically call this local endpoint when `REACT_APP_PLANT_ID_API_KEY` is not set.

Notes
-----
- Use a reasonably balanced dataset with at least a few dozen images per class for decent results.
- Training on GPU is recommended for speed; set up CUDA/PyTorch as needed.
- The training script also supports the existing `api`, `mongo`, and `file` data sources.
