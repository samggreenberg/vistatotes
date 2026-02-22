# VTSearch Architecture

This document is for developers who want to **understand, evaluate, and
selectively extract** components from VTSearch.  It maps the module
structure, dependency graph, and public APIs so you can quickly identify
which pieces you need and how to pull them out.

---

## What VTSearch does

VTSearch is a media-explorer web app for browsing, voting on, and
semantically sorting collections of audio, images, text, or video.  It
combines:

- **Semantic sorting** — LAION-CLAP (audio), CLIP (images), X-CLIP
  (video), E5-base-v2 (text) for embedding-based similarity search.
- **Learned sorting** — a small MLP trained on user votes to predict
  good/bad labels.
- **Flask web UI** — vanilla JS frontend with a REST API.
- **Plugin systems** — auto-discovered dataset importers, results
  exporters, and label importers.

---

## Directory map

```
VTSearch/
├── app.py                          Flask entry point, CLI, startup
├── config.py                       Constants (paths, model IDs, rates)
│
├── vtsearch/
│   ├── media/                      Media type registry + plugins
│   │   ├── base.py                 MediaType ABC, MediaResponse, ProgressCallback
│   │   ├── __init__.py             Registry (register/get/all_types)
│   │   ├── audio/media_type.py     CLAP embeddings
│   │   ├── image/media_type.py     CLIP embeddings
│   │   ├── text/media_type.py      E5 embeddings
│   │   └── video/media_type.py     X-CLIP embeddings
│   │
│   ├── models/                     ML model wrappers
│   │   ├── training.py             MLP training, GMM thresholds (pure PyTorch)
│   │   ├── progress.py             Labelling-progress cache & analysis
│   │   ├── embeddings.py           Thin wrappers around media-type embed()
│   │   └── loader.py               Model initialisation (delegates to media)
│   │
│   ├── datasets/                   Dataset loading & downloading
│   │   ├── origin.py               Origin dataclass (per-element provenance)
│   │   ├── labelset.py             LabelSet / LabeledElement (labeled data with origins)
│   │   ├── loader.py               load_dataset_from_folder/pickle/demo
│   │   ├── downloader.py           HTTP download + ESC-50/CIFAR-10/etc.
│   │   ├── config.py               Demo dataset catalogue
│   │   ├── split.py                Train/test splitting
│   │   └── importers/              Plugin system for data sources
│   │       ├── base.py             DatasetImporter ABC + ImporterField
│   │       ├── folder/             Local directory importer
│   │       ├── pickle/             .pkl file importer
│   │       ├── http_zip/           HTTP archive importer
│   │       ├── rss_feed/           RSS/Podcast feed importer
│   │       └── youtube_playlist/   YouTube yt-dlp importer
│   │
│   ├── exporters/                  Plugin system for output destinations
│   │   ├── base.py                 LabelsetExporter ABC + ExporterField
│   │   ├── file/                   JSON file writer
│   │   ├── csv_file/               CSV file writer
│   │   ├── email_smtp/             SMTP email sender
│   │   ├── webhook/                HTTP POST webhook
│   │   └── gui/                    In-browser display
│   │
│   ├── labels/importers/           Plugin system for label sources
│   │   ├── base.py                 LabelImporter ABC + LabelImporterField
│   │   ├── json_file/              JSON label file reader
│   │   └── csv_file/               CSV label file reader
│   │
│   ├── routes/                     Flask blueprints (HTTP layer)
│   │   ├── clips.py                Clip listing, media serving, voting
│   │   ├── sorting.py              Text/learned/example sort
│   │   ├── detectors.py            Detector export/import/run
│   │   ├── datasets.py             Dataset loading & management
│   │   ├── exporters.py            Exporter registry & execution
│   │   ├── label_importers.py      Label importer registry & execution
│   │   └── main.py                 Root route
│   │
│   ├── utils/
│   │   ├── state.py                Global state (clips, votes, history)
│   │   └── progress.py             Thread-safe progress tracking
│   │
│   ├── audio/                      WAV/tone generation utilities
│   └── cli.py                      CLI autodetect + label-import logic
│
├── static/                         Frontend (HTML + CSS + JS)
└── tests/                          Comprehensive test suite
```

---

## Dependency graph

Arrows point from dependant → dependency.  Modules on the left import
modules on the right.

```
┌──────────────────────────────────────────────────────────┐
│                    Flask / HTTP layer                      │
│                                                          │
│  app.py ──► routes/* ──► utils/state, utils/progress     │
│                │                                          │
│                ├──► models/embeddings, models/training    │
│                ├──► datasets/loader                       │
│                ├──► exporters (registry)                  │
│                └──► labels/importers (registry)           │
└──────────────────────────────────────────────────────────┘
        │               │                │
        ▼               ▼                ▼
┌──────────────┐ ┌────────────┐ ┌───────────────────┐
│ media/*      │ │ models/    │ │ datasets/          │
│              │ │            │ │                     │
│ audio  ─┐   │ │ training   │ │ loader ──► media/*  │
│ image  ─┤   │ │ progress   │ │ downloader          │
│ text   ─┤   │ │ embeddings │ │ importers/*         │
│ video  ─┘   │ │ loader     │ │                     │
│   │         │ │   │        │ │                     │
│   ▼         │ │   ▼        │ └───────────────────┘
│ config.py   │ │ media/*    │
│ torch/HF    │ │ config.py  │
│ (NO Flask)  │ │ (NO Flask) │
└──────────────┘ └────────────┘

┌─────────────────────┐  ┌────────────────────────┐
│ exporters/*         │  │ labels/importers/*      │
│                     │  │                          │
│ base.py (ABC)       │  │ base.py (ABC)           │
│ file, csv, webhook  │  │ json_file, csv_file     │
│ email_smtp, gui     │  │                          │
│                     │  │ (NO Flask, NO state,     │
│ (NO Flask,          │  │  pure data processing)   │
│  NO state,          │  │                          │
│  pure data in/out)  │  └────────────────────────┘
└─────────────────────┘
```

### Key observations

- **media types do NOT import Flask.**  They return a `MediaResponse`
  dataclass; the route layer converts it to a Flask response.
- **models/ do NOT import Flask or global state.**  Functions accept
  `clips_dict`, `good_votes`, `bad_votes` etc. as parameters.
- **exporters and label importers are fully standalone.**  They receive
  a plain dict and return a plain dict/list.  Zero framework coupling.
- **datasets/ functions accept an optional `on_progress` callback.**
  When `None`, they lazily resolve the app's `update_progress`; when
  provided, they use the caller's callback.
- **Only `routes/*` imports global state** from `vtsearch.utils.state`.

---

## Extractability matrix

| Module | Flask? | Global state? | Can extract standalone? |
|--------|--------|---------------|-------------------------|
| `models/training.py` | No | No (params) | **Yes** — pure PyTorch/sklearn |
| `models/progress.py` | No | No (params) | **Yes** — pure torch/numpy |
| `exporters/base.py` + all exporters | No | No | **Yes** — pure data processing |
| `labels/importers/base.py` + all importers | No | No | **Yes** — pure data processing |
| `datasets/downloader.py` | No | No (callback) | **Yes** — requests only |
| `datasets/loader.py` | No | No (callback + params) | **Yes** — needs media registry |
| `datasets/importers/base.py` + all importers | No | No (callback) | **Yes** — each self-contained |
| `media/base.py` | No | No | **Yes** — abstract only |
| `media/audio,image,text,video` | No | No | **Yes** — torch + HF models |
| `utils/progress.py` | No | No | **Yes** — threading only |
| `utils/state.py` | No | N/A (IS the state) | **Yes** — plain Python dicts |
| `config.py` | No | No | **Yes** — just constants |
| `routes/*` | **Yes** | **Yes** | No — Flask-specific |
| `app.py` | **Yes** | **Yes** | No — application entry point |

---

## How to extract specific components

### The ML training pipeline

**Files:** `vtsearch/models/training.py`, `config.py` (for `TRAIN_EPOCHS`)

**Dependencies:** `torch`, `sklearn`, `numpy`

**What you get:** `train_model()` trains a 2-layer MLP classifier on
embeddings + binary labels.  `find_optimal_threshold()` uses a GMM to
pick a decision boundary with configurable FPR/FNR trade-off via an
`inclusion` parameter.

```python
from vtsearch.models.training import train_model, find_optimal_threshold

model = train_model(X_train, y_train, input_dim=512, inclusion_value=0)
threshold = find_optimal_threshold(scores, labels, inclusion_value=0)
```

### Embedding models (CLAP, CLIP, E5, X-CLIP)

**Files:** `vtsearch/media/{audio,image,text,video}/media_type.py`

**Dependencies:** `torch`, `transformers`, `librosa` (audio), `PIL`
(image/video), `sentence-transformers` (text)

Each media type is a self-contained class.  Instantiate it, call
`load_models()`, then use `embed_media()` / `embed_text()`:

```python
from vtsearch.media.audio.media_type import AudioMediaType

audio = AudioMediaType()
audio.load_models()                    # loads CLAP (cached)
embedding = audio.embed_media(Path("example.wav"))  # → numpy array
text_vec  = audio.embed_text("birdsong")            # same space
```

No Flask, no global state, no progress dependency (silent no-op by
default).  To get progress reporting, set a callback before loading:

```python
audio._on_progress = lambda status, msg, cur, tot: print(f"{msg} ({cur}/{tot})")
audio.load_models()
```

### The plugin systems (exporters, importers)

**Pattern:** Each plugin system uses the same architecture:
1. An abstract base class with `fields` (form descriptors) and a
   `run()`/`export()` method.
2. Auto-discovery via `pkgutil.iter_modules` scanning for a sentinel
   attribute (`EXPORTER`, `IMPORTER`, `LABEL_IMPORTER`).
3. CLI support auto-derived from field definitions.

To use an exporter standalone:

```python
from vtsearch.exporters.file import EXPORTER

result = EXPORTER.export(
    results={"media_type": "audio", "results": {...}},
    field_values={"filepath": "/tmp/output.json"},
)
```

### Dataset loading (without Flask)

```python
from vtsearch.datasets.loader import load_dataset_from_folder

clips = {}
load_dataset_from_folder(
    Path("my_audio_folder"),
    media_type="sounds",
    clips=clips,
    on_progress=lambda s, m, c, t: print(f"{m} {c}/{t}"),
)
# clips is now {1: {"id": 1, "embedding": ..., "clip_bytes": ..., ...}, ...}
```

### Progress tracking

**Files:** `vtsearch/utils/progress.py`

A thread-safe progress tracker with no framework dependencies.  Uses
`threading.Lock` and module-level dicts.  Can be dropped into any
application as-is.

---

## Plugin architecture details

All three plugin systems (dataset importers, exporters, label importers)
follow the same pattern:

1. **Base class** defines `name`, `display_name`, `fields`, and an
   abstract `run()`/`export()` method.
2. **Field dataclass** (`ImporterField`, `ExporterField`,
   `LabelImporterField`) describes each user-configurable input with
   type, label, default, validation, and placeholder.
3. **Auto-discovery** scans sub-packages for a sentinel attribute and
   registers them lazily on first access.
4. **CLI support** auto-generates `argparse` flags from field
   definitions.  Override `add_cli_arguments()` for custom handling.
5. **Graceful degradation** — if a plugin's optional dependency is
   missing, a warning is emitted but the app continues.

To add a new plugin, create a package directory, implement the base
class, and expose the sentinel.  See `EXTENDING.md` for full examples.

---

## State management

Application state lives in `vtsearch/utils/state.py` as module-level
dicts:

| Variable | Type | Purpose |
|----------|------|---------|
| `clips` | `dict[int, dict]` | All loaded media clips with embeddings |
| `good_votes` | `dict[int, None]` | Clip IDs voted "good" |
| `bad_votes` | `dict[int, None]` | Clip IDs voted "bad" |
| `label_history` | `list[tuple]` | Ordered labelling events |
| `inclusion` | `int` | FPR/FNR trade-off parameter |
| `favorite_detectors` | `dict` | Saved detector configurations |
| `favorite_extractors` | `dict` | Saved extractor configurations |

**Only Flask routes mutate this state.**  All ML and dataset functions
accept state as parameters — they never import it directly.  This means
you can use the ML code in a script or notebook by passing your own
dicts.

---

## Element-level origin tracking

Every data element (clip) carries its own provenance so that:

- Data from multiple sources can coexist in the same dataset.
- Exported label sets can be re-imported and matched back to their
  original source.
- Origins are preserved through pickle save/load round-trips.

### Per-clip fields

Each clip dict includes two provenance fields:

| Field | Type | Description |
|-------|------|-------------|
| `origin` | `dict \| None` | Serialised `Origin` (e.g. `{"importer": "folder", "params": {"path": "/data"}}`) |
| `origin_name` | `str` | Unique name within the origin (typically the filename) |

### Origin class (`vtsearch/datasets/origin.py`)

```python
from vtsearch.datasets.origin import Origin

o = Origin("folder", {"path": "/data/audio", "media_type": "sounds"})
o.display()   # "folder(/data/audio)"
o.to_dict()   # {"importer": "folder", "params": {"path": "/data/audio", ...}}
```

Origins are set automatically when data is loaded:

- **Importers** produce an `Origin` from their field values via
  `DatasetImporter.build_origin(field_values)`.
- **Demo datasets** get `Origin("demo", {"name": dataset_name})`.
- **Pickle loads** preserve the per-element origins stored in the file.
  Old pickles without origins fall back to the legacy `creation_info` stored in the pickle (if any).

### LabelSet (`vtsearch/datasets/labelset.py`)

A `LabelSet` extends the dataset concept: each element carries its origin,
its name within that origin, **and** its label (`"good"` / `"bad"`).

```python
from vtsearch.datasets.labelset import LabelSet

# Build from current state
ls = LabelSet.from_clips_and_votes(clips, good_votes, bad_votes)

# Build from auto-detect results
ls = LabelSet.from_results(results_dict, clips=clips)

# Serialise / deserialise (superset of legacy label format)
data = ls.to_dict()   # {"labels": [{"md5": ..., "label": ..., "origin": ..., ...}]}
ls2 = LabelSet.from_dict(data)
```

The `GET /api/labels/export` endpoint returns a `LabelSet` serialised
as JSON.  The format is backward-compatible: old consumers that only
read `md5` + `label` continue to work.
