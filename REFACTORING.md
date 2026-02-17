# VistaTotes Refactoring

This document describes the refactored codebase structure for VistaTotes.

## New Package Structure

```
vistatotes/
├── __init__.py           # Package initialization
├── audio/                # Audio utilities
│   ├── __init__.py
│   ├── generator.py      # WAV generation (generate_wav)
│   └── processor.py      # Audio processing (wav_bytes_to_float)
├── datasets/             # Dataset configuration
│   ├── __init__.py
│   └── config.py         # DEMO_DATASETS configuration
├── models/               # ML models and embeddings
│   ├── __init__.py
│   ├── loader.py         # Model initialization (initialize_models)
│   ├── embeddings.py     # Embedding functions for all media types
│   └── training.py       # ML training and scoring
├── routes/               # Flask blueprints (future)
│   └── __init__.py
└── utils/                # Utility modules
    ├── __init__.py
    ├── progress.py       # Progress tracking
    └── state.py          # Global state management

config.py                 # Configuration constants
```

## Migration Guide

### Constants and Configuration

**Before:**
```python
# In app.py
SAMPLE_RATE = 48000
NUM_CLIPS = 20
DATA_DIR = Path("data")
```

**After:**
```python
from config import SAMPLE_RATE, NUM_CLIPS, DATA_DIR
```

### Audio Generation

**Before:**
```python
def generate_wav(frequency: float, duration: float) -> bytes:
    # ... implementation in app.py
```

**After:**
```python
from vistatotes.audio import generate_wav, wav_bytes_to_float
```

### Model Loading

**Before:**
```python
def initialize_app():
    global clap_model, clap_processor
    # ... load models
```

**After:**
```python
from vistatotes.models import initialize_models, get_clap_model

initialize_models()
clap_model, clap_processor = get_clap_model()
```

### Embeddings

**Before:**
```python
def embed_audio_file(audio_path: Path) -> Optional[np.ndarray]:
    if clap_model is None or clap_processor is None:
        return None
    # ... implementation
```

**After:**
```python
from vistatotes.models import embed_audio_file, embed_text_query

# Embed media files
embedding = embed_audio_file(audio_path)

# Embed text queries
text_embedding = embed_text_query("dog barking", media_type="audio")
```

### Training and Scoring

**Before:**
```python
def train_and_score(inclusion_value=0):
    X_list = []
    for cid in good_votes:
        X_list.append(clips[cid]["embedding"])
    # ...
```

**After:**
```python
from vistatotes.models import train_and_score
from vistatotes.utils import clips, good_votes, bad_votes, get_inclusion

results, threshold = train_and_score(clips, good_votes, bad_votes, get_inclusion())
```

### Progress Tracking

**Before:**
```python
def update_progress(status, message="", current=0, total=0, error=None):
    with progress_lock:
        progress_data["status"] = status
        # ...
```

**After:**
```python
from vistatotes.utils import update_progress, get_progress

update_progress("loading", "Loading models...", 1, 3)
progress = get_progress()
```

### State Management

**Before:**
```python
clips = {}
good_votes = {}
bad_votes = {}
inclusion = 0
```

**After:**
```python
from vistatotes.utils import clips, good_votes, bad_votes
from vistatotes.utils import clear_all, set_inclusion, get_inclusion

# Use imported state
clips[clip_id] = {...}
good_votes[clip_id] = None

# Clear state
clear_all()

# Manage inclusion
set_inclusion(5)
current_inclusion = get_inclusion()
```

### Dataset Configuration

**Before:**
```python
DEMO_DATASETS = {
    "animals": {...},
    # ...
}
```

**After:**
```python
from vistatotes.datasets import DEMO_DATASETS

for dataset_name, config in DEMO_DATASETS.items():
    print(f"{dataset_name}: {config['description']}")
```

## Benefits of Refactoring

1. **Modularity**: Code is organized into logical, reusable modules
2. **Maintainability**: Easier to find and update specific functionality
3. **Testability**: Each module can be tested independently
4. **Scalability**: Easier to add new features without bloating main file
5. **Clarity**: Clear separation of concerns (models, audio, datasets, utils)

## Next Steps

1. Gradually migrate `app.py` to import from new modules
2. Create Flask blueprints for routes (clips, votes, sorting, datasets)
3. Split tests into module-specific test files
4. Add docstrings and type hints throughout
5. Create integration tests for the full workflow

## Backward Compatibility

The refactored modules can be adopted incrementally:
- All functions maintain the same signatures
- Global state is still accessible (though now properly encapsulated)
- No breaking changes to the API
