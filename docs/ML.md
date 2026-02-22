# Machine Learning Details

VTSearch uses a small MLP (multi-layer perceptron) neural network to learn a binary classifier from user votes ("good" vs "bad"). The model operates on embeddings produced by pretrained feature extractors and outputs a score in [0, 1] for each item in the dataset.

## Architecture

The MLP is defined in `vtsearch/models/training.py` via `build_model()`:

```
Linear(input_dim, 64) -> ReLU -> Linear(64, 1)
```

- **Input dimension**: Dynamic, depends on the embedding model for the current media type (see [Embedding Models](#embedding-models) below).
- **Hidden layer**: 64 neurons with ReLU activation.
- **Output**: A single logit. `torch.sigmoid` is applied at inference time to produce a probability in [0, 1].

The model outputs raw logits (not probabilities) during training. This allows the use of `BCEWithLogitsLoss`, which fuses the sigmoid and binary cross-entropy computation using the log-sum-exp trick for better numerical stability. At inference time, `torch.sigmoid()` is applied explicitly to convert logits to probabilities.

## Training Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Loss function** | `BCEWithLogitsLoss` | Per-sample (unreduced), with manual class weighting |
| **Optimizer** | Adam | `lr=0.001`, `weight_decay=1e-4` |
| **Epochs** | 200 | Configurable via `TRAIN_EPOCHS` in `config.py` |
| **Batching** | Full-batch | All labeled data in every forward pass |
| **Reproducibility** | `torch.manual_seed(42)` | Set before model construction for deterministic weight init |
| **Gradient scoping** | `torch.enable_grad()` | Explicitly enabled during training loop |

### Class Weighting

Training uses inverse-frequency weighting to balance classes, with an additional `inclusion_value` parameter (range [-10, +10]) that lets users bias the model toward recall or precision:

- **Base weights**: `weight_true = num_false / num_true`, `weight_false = 1.0`
- **Inclusion >= 0**: `weight_true *= 2^inclusion_value` (include more items)
- **Inclusion < 0**: `weight_false *= 2^(-inclusion_value)` (exclude more items)

### Threshold Calibration

A decision threshold separating "good" from "bad" predictions is computed via **cross-calibration**:

1. Split labeled data into two halves (D1, D2).
2. Train model M1 on D1, find optimal threshold t1 by evaluating M1 on D2.
3. Train model M2 on D2, find optimal threshold t2 by evaluating M2 on D1.
4. Return `(t1 + t2) / 2`.

The optimal threshold at each split minimizes a weighted combination of false-positive rate and false-negative rate, governed by the same `inclusion_value`.

For semantic (text/example) sorts, a **GMM-based threshold** is used instead: a 2-component Gaussian Mixture Model is fitted to the score distribution and the midpoint between component means serves as the threshold.

## PyTorch Environment Settings

| Setting | Where | Value |
|---------|-------|-------|
| `OMP_NUM_THREADS` | `app.py` | `1` |
| `MKL_NUM_THREADS` | `app.py` | `1` |
| `torch.set_num_threads` | `vtsearch/models/loader.py` | `1` |
| dtype | `training.py` | `torch.float32` |
| Device | default | CPU (GPU supported, see tests) |

Threading is restricted to 1 to minimize memory overhead — the real cost is the embedding models, not the MLP.

## Embedding Models

Each media type uses a different pretrained model to produce fixed-size embedding vectors:

| Media type | Model | Embedding dim |
|------------|-------|--------------|
| Audio | LAION CLAP (`laion/clap-htsat-unfused`) | 512 |
| Image | OpenAI CLIP (`openai/clip-vit-base-patch32`) | 512 |
| Video | Microsoft X-CLIP (`microsoft/xclip-base-patch32`) | 768 |
| Text | E5 (`intfloat/e5-base-v2`) | 768 |

Embeddings are computed once when a dataset is loaded and stored as `numpy.ndarray` in each clip's `"embedding"` field. The MLP trains on these pre-computed vectors, so training is fast (typically < 1 second for 200 epochs on a few hundred labeled examples).

## Model Serialization

Trained models are serialized as JSON dictionaries mapping state_dict keys to nested lists:

```json
{
    "0.weight": [[...], ...],
    "0.bias": [...],
    "2.weight": [[...]],
    "2.bias": [...]
}
```

To reconstruct a model from saved weights, use `build_model(input_dim)` followed by `load_state_dict()`. The `input_dim` can be inferred from the first layer weights: `len(weights["0.weight"][0])`.

## Key Files

- `vtsearch/models/training.py` — `build_model`, `train_model`, `train_and_score`, threshold functions
- `vtsearch/models/progress.py` — Cached per-step training and stability analysis
- `vtsearch/models/loader.py` — Model initialization and thread configuration
- `vtsearch/eval/voting_iterations.py` — Voting simulation evaluation
- `config.py` — `TRAIN_EPOCHS` and model IDs
