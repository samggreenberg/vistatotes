# CLAUDE.md - VistaTotes Project Context

## Project Overview
**VistaTotes** is a media explorer web app for browsing and voting on audio clips, images, or text paragraphs. It supports semantic sorting (via embeddings: LAION-CLAP, CLIP, E5) and learned sorting (neural network trained on votes).

**Tech Stack:**
- Backend: Flask (Python 3.9+)
- Frontend: Vanilla JavaScript (single-page app)
- ML: PyTorch for embeddings and model training
- Storage: pickle files, in-memory state

## Project Structure
```
vistatotes/
├── app.py                      # Flask entry point (routes registration, startup)
├── config.py                   # Constants: paths, URLs, model IDs, sample rates
├── static/
│   └── index.html              # Single-page frontend (HTML/CSS/vanilla JS)
├── test_app.py                 # pytest test suite
├── vistatotes/                 # Main package
│   ├── routes/                 # Flask blueprints (main, clips, sorting, datasets)
│   ├── models/                 # ML models: embeddings, training, model loading
│   ├── audio/                  # Audio: WAV generation, processing
│   ├── datasets/               # Dataset loading: config, downloader, importers
│   │   └── importers/          # folder, pickle, http_zip importers
│   ├── media/                  # Media handling (images, video, text)
│   └── utils/                  # State management, progress tracking
├── requirements.txt            # Core dependencies
├── requirements-{cpu,gpu}.txt  # PyTorch variants
├── requirements-dev.txt        # Dev tools (pytest)
└── requirements-importers.txt  # Optional importers
```

## Key Modules & Responsibilities
- **app.py**: Initializes Flask, registers blueprints, handles startup (init_clips, initialize_models)
- **config.py**: Audio settings (SAMPLE_RATE=48000, NUM_CLIPS=20), paths, dataset URLs, model IDs
- **vistatotes/routes/**: Flask blueprints handling API endpoints
- **vistatotes/models/**: Embedding models (CLAP, CLIP, E5), model training, progress tracking
- **vistatotes/datasets/**: Dataset management, downloading, importing from various sources
- **vistatotes/audio/**: Synthetic WAV generation, audio processing
- **vistatotes/media/**: Media type handling (images, text, video)
- **vistatotes/utils/**: Global state (clips dict), progress bar utilities

## Setup & Running
```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-cpu.txt  # or requirements-gpu.txt
python app.py              # Production mode (loads models, generates clips)
python app.py --local      # Local dev mode (models lazy-loaded)
```

## Testing
```bash
pip install -r requirements-dev.txt
python -m pytest test_app.py -v
```

## Important Notes
- **Data directory**: `data/` is created at runtime; contains embeddings, models cache, audio/images
- **Embeddings**: LAION-CLAP for audio, CLIP for images, E5 for text
- **Frontend state**: Stateless; votes/selections managed via Flask backend
- **Demo datasets**: ESC-50 (audio), CIFAR-10 (images), 20 Newsgroups (text) — downloaded on first use and cached
- **Startup time**: Model loading can take time; startup feedback via tqdm progress bars
- **Environment**: OMP_NUM_THREADS and MKL_NUM_THREADS set to 1 (memory constraint optimization)

## Common Tasks Quick Reference
| Task | Command |
|------|---------|
| Start app | `python app.py` |
| Start local dev | `python app.py --local` |
| Run tests | `python -m pytest test_app.py -v` |
| View test coverage | See TEST_COVERAGE_ANALYSIS.md |
| Install GPU support | `pip install -r requirements-gpu.txt` |

## Frontend Basics
- Single HTML file: `static/index.html`
- Vanilla JS (no framework)
- Interacts with Flask API endpoints registered in `vistatotes/routes/`
- Supports: play audio, view images, read text, vote (good/bad), change sorting method

## Recent Changes
See REFACTORING.md for recent architectural improvements and ongoing modernization efforts.

## Development Branch
Use branch format: `claude/create-claude-md-YY0ev` (provided in task context)
