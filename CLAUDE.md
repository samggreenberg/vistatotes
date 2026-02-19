# VistaTotes

Media explorer web app for browsing/voting on audio, images, or text. Semantic sorting (LAION-CLAP, CLIP, E5 embeddings) and learned sorting (neural net trained on votes). Flask + vanilla JS + PyTorch.

## Commands
- **Run tests**: `python -m pytest test_app.py -v`
- **Start app**: `python app.py` (or `python app.py --local` for dev)
- **Install deps**: `pip install -r requirements-cpu.txt` (or `requirements-gpu.txt`)

## Architecture
- `app.py` — Flask entry point, registers blueprints, startup logic
- `config.py` — Constants (SAMPLE_RATE, NUM_CLIPS, paths, model IDs)
- `vistatotes/routes/` — Flask blueprints: `main.py`, `clips.py`, `sorting.py`, `datasets.py`
- `vistatotes/models/` — Embeddings, training, model loading, progress tracking
- `vistatotes/datasets/` — Dataset loading, downloading, importers (folder/pickle/http_zip)
- `vistatotes/media/` — Media type plugins: audio, image, text, video
- `vistatotes/utils/` — Global state (`clips` dict, votes), progress utilities
- `static/index.html` — Entire frontend (HTML/CSS/JS, 2700 lines — read selectively)
- `test_app.py` — Full test suite (1500 lines — read selectively)

## Key Details
- Global state lives in `vistatotes/utils/state.py`: `clips`, `good_votes`, `bad_votes` are module-level dicts
- Votes are `dict[int, None]` (not sets) — use `votes[id] = None` syntax
- `data/` dir created at runtime for embeddings, model cache, media files
- OMP_NUM_THREADS and MKL_NUM_THREADS set to 1 for memory optimization
- Linter: ruff (E402 ignored, see pyproject.toml)

## Large Files Warning
These files are expensive to read in full. Prefer reading specific line ranges:
- `static/index.html` (2678 lines) — CSS is lines 1-400, JS starts ~line 450
- `test_app.py` (1532 lines) — use grep to find specific test classes
- `vistatotes/datasets/loader.py` (937 lines)
- `vistatotes/routes/sorting.py` (816 lines)
