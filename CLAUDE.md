# VistaTotes

Media explorer web app for browsing/voting on audio, images, or text. Semantic sorting (LAION-CLAP, CLIP, E5 embeddings) and learned sorting (neural net trained on votes). Flask + vanilla JS + PyTorch.

## Commands
- **Run tests**: `python -m pytest tests/ -v`
- **Start app**: `python app.py` (or `python app.py --local` for dev)
- **CLI autodetect**: `python app.py --autodetect --dataset <file.pkl> --detector <file.json>`
- **CLI autodetect + exporter**: `python app.py --autodetect --dataset <file.pkl> --detector <file.json> --exporter file --filepath results.json`
- **Install deps**: `pip install -r requirements-cpu.txt` (or `requirements-gpu.txt`)
- **Lint**: `ruff check .`
- **Format**: `ruff format .`

## Architecture
- `app.py` — Flask entry point, registers blueprints, startup logic, CLI argument parsing
- `vistatotes/cli.py` — CLI autodetect: load dataset + detector, run inference, export results
- `config.py` — Constants (SAMPLE_RATE, NUM_CLIPS, paths, model IDs)
- `vistatotes/routes/` — Flask blueprints: `main.py`, `clips.py`, `sorting.py`, `datasets.py`
- `vistatotes/models/` — Embeddings, training, model loading, progress tracking
- `vistatotes/datasets/` — Dataset loading, downloading, importers (folder/pickle/http_zip)
- `vistatotes/media/` — Media type plugins: audio, image, text, video
- `vistatotes/utils/` — Global state (`clips` dict, votes), progress utilities
- `static/index.html` — HTML structure (270 lines)
- `static/styles.css` — All CSS styles
- `static/app.js` — All frontend JavaScript
- `tests/` — Test suite split by module:
  - `conftest.py` — Shared fixtures (client, vote reset, model init)
  - `test_audio.py` — WAV generation
  - `test_clips.py` — Clip init, listing, audio endpoint, MD5
  - `test_votes.py` — Voting and vote retrieval
  - `test_sorting.py` — Text sort, learned sort, example sort, train_and_score
  - `test_labels.py` — Label export/import
  - `test_inclusion.py` — Inclusion GET/POST
  - `test_detectors.py` — Detector export, detector sort, favorites, auto-detect
  - `test_cli_autodetect.py` — CLI autodetect: run_autodetect function, --autodetect flag, --exporter flag
  - `test_datasets.py` — Dataset endpoints, startup state, importers, archive extraction

## Key Details
- Global state lives in `vistatotes/utils/state.py`: `clips`, `good_votes`, `bad_votes` are module-level dicts
- Votes are `dict[int, None]` (not sets) — use `votes[id] = None` syntax
- `data/` dir created at runtime for embeddings, model cache, media files
- OMP_NUM_THREADS and MKL_NUM_THREADS set to 1 for memory optimization
- Linter/formatter: ruff (E402 ignored, line-length 120, see pyproject.toml)
