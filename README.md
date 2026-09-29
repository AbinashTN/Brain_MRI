# Educational brain MRI classifier

A small transfer-learning project using MobileNetV3 Small, PyTorch and Streamlit.
The four classes are `glioma`, `meningioma`, `notumor` and `pituitary`.
Pedictions are not medical diagnoses.

## Dataset

The dataset was found on Kaggle and can be downloaded here:
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## Setup

Run commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the project

Edit `config.py` to set the dataset and output paths, number of epochs, batch
size, learning rate, validation proportion and random seed. Training, evaluation,
exploration, Streamlit and the API share this configuration. Run the scripts
without command-line options; no configuration loader or extra dependency is needed.
Set `API_URL` to the FastAPI server address (default: `http://127.0.0.1:8000`).
Restart Streamlit or the API after changing the configuration.

```bash
# Optional exploration: class counts, examples and image sizes.
python -m scripts.explore_data

# Train and keep the model with the lowest validation loss.
python -m src.train

# Evaluate on Testing.
python -m src.evaluate

# Start the prediction API (requires the saved model).
python -m uvicorn api.main:app --reload

# In a second terminal, start the Streamlit interface.
python -m streamlit run app/streamlit_app.py

# Run tests without pretrained downloads or the real dataset.
python -m pytest -q
```

Streamlit sends the selected image to the API with `POST /predict` and displays
the JSON response. Only the API loads the model for the interface; both servers
must be running. The API exposes `/health`, `/predict` and interactive documentation
at `/docs`.
Training downloads ImageNet weights on its first run if needed and uses CUDA
when available. Inference and evaluation run on CPU for simplicity.

## How it works

1. `load_train_test()` returns `train_set` and `test_set` as lists of `(path, label)`.
   Only `*.jpg` files are read. Identical decoded RGB images are kept once,
   including across Training and Testing; the Training copy takes priority.
   Files on disk are never deleted. Unreadable images are skipped with an
   English error log. An empty usable split or identical images with different
   labels raise a clear error.
2. Scikit-learn's `train_test_split` divides Training into 80% training and
   20% validation, preserving class proportions with `stratify` and seed 42.
   Testing is used only for final evaluation. Exact deduplication does not
   detect similar slices or images from the same patient.
3. MobileNet's ImageNet preset resizes to 256, center-crops to 224 and normalizes
   RGB inputs. Training adds a small random rotation.
4. The feature extractor and its BatchNorm statistics remain frozen. Adam
   trains the classifier with learning rate 0.001 for the requested epochs.
   The best validation loss selects the saved model; there is no early stopping.
5. Cross-entropy uses integer labels `0, 1, 2, 3`. One-hot encoding is unnecessary
   for a single class per image. Softmax converts outputs into scores for display;
   these scores are not calibrated medical probabilities.

The code intentionally uses a fixed model, class order and preprocessing.
There are no split manifests, audit JSON files or configurable worker settings.
`DataLoader` is called directly where batches are needed.

## Files and outputs

| File                      | Purpose                                           |
| ------------------------- | ------------------------------------------------- |
| `config.py`               | Shared paths and training settings                |
| `src/dataset.py`          | Read JPG images, exclude duplicates, load tensors |
| `src/preprocessing.py`    | Shared ImageNet transform                         |
| `src/model.py`            | Build MobileNet and freeze features               |
| `src/train.py`            | Split, train, validate and save the best model    |
| `src/inference.py`        | Load a model and predict one image                |
| `src/evaluate.py`         | Classification report, confusion matrix, examples |
| `src/utils.py`            | Set random seeds and write JSON                   |
| `scripts/explore_data.py` | Plot the dataset after duplicate exclusion        |
| `app/streamlit_app.py`    | Upload an image or select an example              |
| `api/main.py`             | HTTP inference used by Streamlit                  |

Generated outputs:

- `models/best_model.pth`: model weights and class names.
- `outputs/figures/`: exploration, training curves, confusion matrix and predictions.
- `outputs/metrics/test_metrics.json` and `classification_report.txt`: test results.

Existing checkpoints using the same MobileNet and default 224-pixel preprocessing
remain loadable. Restart the API after retraining to load the new model weights.
Old reports from the previous code may still exist; they are not used by this code.
Rerun training and evaluation to produce results for the current pipeline.

Tests use small temporary JPG datasets and randomly initialized weights. They
check duplicate exclusion, transforms, frozen features, a classifier update,
checkpoint loading, the configured pipeline and HTTP predictions. They do not
measure model quality.
