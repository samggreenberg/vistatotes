# VTSearch

Media explorer web app for browsing/voting on audio, images, or text. Semantic sorting (LAION-CLAP, CLIP, E5 embeddings) and learned sorting (neural net trained on votes). Flask + vanilla JS + PyTorch.

## Commands
- **Run tests (CPU, fast)**: `bash .claude/hooks/ensure-test-deps.sh && python -m pytest tests/ -v`
- **Run tests (CPU, full)**: `bash .claude/hooks/ensure-test-deps.sh && python -m pytest tests/ -v -m 'not gpu'`
- **Run slow CLI subprocess tests only**: `python -m pytest tests/ -v -m slow`
- **Run GPU tests**: `python -m pytest tests/test_gpu.py -v -m gpu` (requires CUDA GPU; downloads models on first run)
- **Run all tests (CPU + GPU)**: `python -m pytest tests/ -v -m ''`
- **Start app**: `bash .claude/hooks/ensure-test-deps.sh && python app.py` (or `python app.py --local` for dev)
- **CLI autodetect**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json>`
- **CLI autodetect + exporter**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json> --exporter file --filepath results.json`
- **CLI label import**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --import-labels --dataset <file.pkl> --label-importer json_file --file labels.json`
- **CLI label import (CSV)**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --import-labels --dataset <file.pkl> --label-importer csv_file --file labels.csv`
- **CLI processor import**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --import-processor --processor-importer detector_file --processor-name "my detector" --file detector.json`
- **Install deps**: `pip install -r requirements-cpu.txt` (or `requirements-gpu.txt`)
- **Lint**: `ruff check .`
- **Format**: `ruff format .`

## Architecture
- `app.py` — Flask entry point, registers blueprints, startup logic, CLI argument parsing
- `vtsearch/cli.py` — CLI utilities: autodetect (load dataset + detector, run inference, export results), import_labels_main (load dataset + label importer), import_processor_main (import detector via processor importer)
- `config.py` — Constants (SAMPLE_RATE, NUM_CLIPS, paths, model IDs)
- `vtsearch/routes/` — Flask blueprints: `main.py`, `clips.py`, `sorting.py`, `detectors.py`, `datasets.py`, `exporters.py`, `label_importers.py`, `processor_importers.py`
- `vtsearch/models/` — Embeddings, training, model loading, progress tracking
- `vtsearch/datasets/` — Dataset loading, downloading, origin tracking, labelsets, importers (folder/pickle/http_zip/rss_feed/youtube_playlist)
- `vtsearch/exporters/` — Results exporters (file/gui/email_smtp/csv_file/webhook)
- `vtsearch/labels/importers/` — Label importers (json_file/csv_file); auto-discovered via `LABEL_IMPORTER` sentinel
- `vtsearch/processors/importers/` — Processor importers (detector_file/label_file); auto-discovered via `PROCESSOR_IMPORTER` sentinel
- `vtsearch/media/` — Media type plugins: audio, image, text, video
- `vtsearch/utils/` — Global state (`clips` dict, votes), progress utilities
- `static/index.html` — HTML structure (270 lines)
- `static/styles.css` — All CSS styles
- `static/app.js` — All frontend JavaScript
- `tests/` — Test suite split by module:
  - `conftest.py` — Shared fixtures (client, vote reset, model init)
  - `test_audio.py` — WAV generation
  - `test_clips.py` — Clip init, listing, audio endpoint, MD5
  - `test_votes.py` — Voting and vote retrieval
  - `test_sorting.py` — Text sort, learned sort, example sort, train_and_score
  - `test_labels.py` — Label export/import (via /api/labels/export and /api/labels/import)
  - `test_label_importers.py` — Label importer base class, registry, json_file/csv_file importers, API routes
  - `test_inclusion.py` — Inclusion GET/POST
  - `test_detectors.py` — Detector export, detector sort, favorites, auto-detect
  - `test_cli_autodetect.py` — CLI autodetect: run_autodetect function, --autodetect flag, --exporter flag. Subprocess tests marked `slow` (~16s each, excluded from default run)
  - `test_datasets.py` — Dataset endpoints, startup state, importers, archive extraction
  - `test_rss_youtube_importers.py` — RSS feed and YouTube playlist importer metadata, CLI args, run logic
  - `test_csv_webhook_exporters.py` — CSV and Webhook exporter metadata, CLI args, export logic
  - `test_exporters.py` — Results exporter base classes, registry, built-in exporters, API routes
  - `test_importers.py` — Importer base class, HTTP archive/folder importer metadata, archive extraction
  - `test_extractors.py` — Image class extractor
  - `test_processors.py` — Media processor tests
  - `test_processor_importers.py` — Processor importer base class, registry, detector_file/label_file importers, API routes, CLI
  - `test_origin_labelset.py` — Origin class, LabeledElement, LabelSet, build_origin(), label export/import with origins, integration
  - `test_gpu.py` — GPU tests: training, cross-calibration, detectors, embedding models (CLAP/CLIP/X-CLIP/E5), CPU↔GPU equivalence, memory cleanup (skipped without CUDA)

## Test Markers
- **Default** (`pytest tests/ -v`): Runs fast CPU tests only (~35s). Excludes `gpu` and `slow` markers.
- **`slow`**: CLI subprocess tests that spawn `python app.py --autodetect` (each ~16s, total ~290s). Run with `-m slow` or include with `-m 'not gpu'`.
- **`gpu`**: CUDA-only tests. Run with `-m gpu`.
- **All tests**: Use `-m ''` to run everything.

## Test Workflow (IMPORTANT)

Testing can crash the session. To avoid losing work, follow this workflow:

1. **Commit and push before running tests.** Before running `pytest` or any test command, commit all current changes and push to your working branch. Use a message like `"WIP: pre-test checkpoint"` if the work isn't finalized yet.
2. **Run tests.**
3. **If tests fail and fixes are needed**, make the fixes, then commit and push again before re-running tests.
4. **Repeat** until tests pass. Every cycle of fixes should be committed and pushed before the next test run.

This ensures work is recoverable if the session crashes during a test run.

## Key Details
- Global state lives in `vtsearch/utils/state.py`: `clips`, `good_votes`, `bad_votes` are module-level dicts
- Votes are `dict[int, None]` (not sets) — use `votes[id] = None` syntax
- Each clip has `origin` (dict or None) and `origin_name` (str) for per-element provenance tracking
- `Origin` class in `vtsearch/datasets/origin.py`; `LabelSet`/`LabeledElement` in `vtsearch/datasets/labelset.py`
- Label export (`/api/labels/export`) returns a `LabelSet` with per-element origin info (superset of legacy format)
- `data/` dir created at runtime for embeddings, model cache, media files
- OMP_NUM_THREADS and MKL_NUM_THREADS set to 1 for memory optimization
- Linter/formatter: ruff (E402 ignored, line-length 120, see pyproject.toml)
