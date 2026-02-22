# Extending VTSearch

This guide explains how to add a new **Data Importer**, **Results Exporter**, or
**Media Type** to VTSearch.  Each section describes the interface contract,
where files go, and how to wire up dependencies.

---

## Table of Contents

1. [Adding a Data Importer](#adding-a-data-importer)
2. [Adding a Results Exporter](#adding-a-results-exporter)
3. [Adding a Media Type](#adding-a-media-type)
4. [Dependency Management (requirements.txt)](#dependency-management)

---

## Adding a Data Importer

Data importers let users load datasets from new sources (S3 buckets, databases,
APIs, etc.).  The system auto-discovers importers at runtime — no changes to
routes or core code are needed.

### How discovery works

The registry in `vtsearch/datasets/importers/__init__.py` uses `pkgutil` to
scan for **sub-packages** (directories with `__init__.py`) under
`vtsearch/datasets/importers/`.  For each sub-package it finds, it imports the
module and looks for a module-level attribute named `IMPORTER`.  If that
attribute exists and is a `DatasetImporter` instance, it is registered
automatically.

Failed imports emit a warning but do not break the application — this means a
missing optional dependency gracefully disables that importer rather than
crashing the whole app.

### File structure

Create a new sub-package directory:

```
vtsearch/datasets/importers/<your_importer>/
├── __init__.py       # Importer class + IMPORTER instance (required)
└── requirements.txt  # Pip dependencies, even if empty (required)
```

### What to implement

Subclass `DatasetImporter` from `vtsearch.datasets.importers.base` and set the
required class attributes.  Then implement the `run()` method and expose a
module-level `IMPORTER` instance.

```python
# vtsearch/datasets/importers/s3/__init__.py

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import load_dataset_from_folder


class S3Importer(DatasetImporter):
    # --- Required class attributes ---

    name = "s3"
    #   Internal snake_case identifier.  Used in API routes:
    #   POST /api/dataset/import/s3

    display_name = "AWS S3 Bucket"
    #   Human-readable label shown in the frontend UI.

    description = "Download media files from an S3 bucket."
    #   One-sentence subtitle shown below the display name.

    icon = "☁️"
    #   Emoji shown next to the display name.  Defaults to "🔌" if omitted.

    fields = [
        ImporterField(
            key="bucket",          # Form field identifier
            label="Bucket Name",   # UI label
            field_type="text",     # One of: "text", "url", "folder", "file", "select"
            description="The S3 bucket name.",
            required=True,         # Defaults to True
        ),
        ImporterField(
            key="prefix",
            label="Key Prefix",
            field_type="text",
            description="Optional prefix to filter objects.",
            required=False,
            default="",
        ),
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            options=["sounds", "videos", "images", "paragraphs"],
            default="sounds",
        ),
    ]

    def run(self, field_values: dict, clips: dict) -> None:
        """Download files from S3, then load them into the dataset.

        Args:
            field_values: Maps each ImporterField.key to the user's input.
                - "file" fields arrive as werkzeug FileStorage objects.
                - All other fields arrive as plain strings.
            clips: The global clips dict.  Populate it **in-place**; do not
                replace the reference.
        """
        import boto3
        from pathlib import Path
        from config import DATA_DIR
        from vtsearch.utils import update_progress

        bucket = field_values["bucket"]
        prefix = field_values.get("prefix", "")
        media_type = field_values.get("media_type", "sounds")

        # Download files to a local temp directory
        download_dir = DATA_DIR / "s3_import"
        download_dir.mkdir(parents=True, exist_ok=True)

        s3 = boto3.client("s3")
        objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys = [o["Key"] for o in objects.get("Contents", [])]

        for i, key in enumerate(keys):
            update_progress("downloading", f"Downloading {key}", i, len(keys))
            local_path = download_dir / Path(key).name
            s3.download_file(bucket, key, str(local_path))

        # Delegate to the standard folder loader
        load_dataset_from_folder(download_dir, media_type, clips)


# This module-level instance is what the registry discovers.
IMPORTER = S3Importer()
```

### Element-level origin tracking

Every clip produced by an importer is automatically tagged with an
**origin** — a dict identifying the importer and its parameters.  This
happens in `_run_importer_in_background()` after `run()` completes:

```python
clip["origin"]      = {"importer": "s3", "params": {"bucket": "my-data"}}
clip["origin_name"] = "clip_001.wav"  # defaults to clip["filename"]
```

If your importer pre-populates `clip["origin"]` in `run()`, those values
are preserved.  Otherwise the system calls `build_origin(field_values)` on
your importer class (inherited from `DatasetImporter`) and applies the
result to all clips that lack an origin.

Origins enable per-element provenance in label exports, results exports,
and pickle round-trips.

### Class attributes reference

| Attribute      | Type                | Required | Description                                          |
|----------------|---------------------|----------|------------------------------------------------------|
| `name`         | `str`               | Yes      | Snake_case identifier, used in API URL path          |
| `display_name` | `str`               | Yes      | Human-readable label for the UI                      |
| `description`  | `str`               | Yes      | One-sentence subtitle                                |
| `icon`         | `str`               | No       | Emoji/icon string (default `"🔌"`)                   |
| `fields`       | `list[ImporterField]`| Yes     | Ordered list of user-facing input fields             |

### ImporterField options

| Parameter    | Type        | Default  | Description                                            |
|--------------|-------------|----------|--------------------------------------------------------|
| `key`        | `str`       | —        | Field identifier (used as dict key in `field_values`)  |
| `label`      | `str`       | —        | Display label in the UI                                |
| `field_type` | `FieldType` | —        | `"text"`, `"url"`, `"folder"`, `"file"`, or `"select"` |
| `description`| `str`       | `""`     | Helper text shown below the field                      |
| `accept`     | `str`       | `""`     | For `"file"` fields: comma-separated extensions (e.g. `".pkl"`) |
| `options`    | `list[str]` | `[]`     | For `"select"` fields: allowed dropdown values         |
| `default`    | `str`       | `""`     | Pre-filled value                                       |
| `required`   | `bool`      | `True`   | Whether the field must be filled before importing      |

### How it gets invoked

1. The frontend calls `GET /api/dataset/importers` to discover available
   importers.  The response includes your importer's `name`, `display_name`,
   `description`, `icon`, and `fields` — the UI renders a form automatically.
2. The user fills out the form and submits it.  The frontend sends
   `POST /api/dataset/import/<name>` with either a JSON body or
   `multipart/form-data` (if any field has `field_type="file"`).
3. The route handler in `vtsearch/routes/datasets.py` extracts `field_values`
   from the request, clears the current dataset, and calls `importer.run()`
   in a background daemon thread.
4. The frontend polls `GET /api/dataset/progress` to show a progress bar.

### Progress reporting

Call `update_progress()` from your `run()` method to give the user feedback:

```python
from vtsearch.utils import update_progress

update_progress("downloading", "Downloading file 3/10", 3, 10)
#                ^status        ^message              ^cur ^total
```

Any unhandled exception in `run()` is caught by the background thread wrapper
and stored as an error in the progress tracker.

### Wiring up dependencies

1. Create `vtsearch/datasets/importers/<name>/requirements.txt` listing any
   pip packages your importer needs (create the file even if empty — see
   [Dependency Management](#dependency-management) below).
2. Add a reference line to `requirements-importers.txt`:
   ```
   -r vtsearch/datasets/importers/<name>/requirements.txt
   ```
3. If using `requirements-cpu.txt`, add the packages inline in that file too
   (it uses inline pins instead of `-r` includes for version control).

---

## Adding a Results Exporter

VTSearch currently exports three kinds of results, each served from an
existing API endpoint.  There is no abstract base class or auto-discovery
mechanism for exporters (unlike importers).  Adding a new exporter means adding
a new route to the appropriate blueprint.

### Existing export endpoints

| Endpoint                  | Method | What it exports                              | Format          | Blueprint    |
|---------------------------|--------|----------------------------------------------|-----------------|--------------|
| `/api/dataset/export`     | GET    | Full dataset (clips + embeddings + media)    | Pickle (`.pkl`) | `datasets_bp`|
| `/api/labels/export`      | GET    | LabelSet — labels with per-element origin    | JSON            | `sorting_bp` |
| `/api/detector/export`    | POST   | Trained MLP weights + threshold              | JSON            | `sorting_bp` |

### Label export format (LabelSet)

The label export endpoint (`GET /api/labels/export`) returns a
`LabelSet` — each entry includes the element's origin and name in
addition to its MD5 and label.  This is a superset of the legacy format:

```json
{
  "labels": [
    {
      "md5": "d41d8cd98f...",
      "label": "good",
      "origin": {"importer": "folder", "params": {"path": "/data/audio"}},
      "origin_name": "bark_001.wav",
      "filename": "bark_001.wav",
      "category": "dogs"
    }
  ],
}
```

Old consumers that only read `md5` and `label` continue to work unchanged.

### Where to put a new exporter

Decide which blueprint your exporter belongs to based on what it exports:

- **Dataset-level exports** (clip data, metadata) → `vtsearch/routes/datasets.py` on `datasets_bp`
- **Sorting / voting / model exports** (labels, detectors, scores) → `vtsearch/routes/sorting.py` on `sorting_bp`

### Example: CSV results exporter

```python
# Add to vtsearch/routes/sorting.py (or datasets.py)

import csv
import io

@sorting_bp.route("/api/results/export-csv")
def export_results_csv():
    """Export clip scores and votes as a CSV file."""
    if not clips:
        return jsonify({"error": "No dataset loaded"}), 400

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["clip_id", "filename", "md5", "category", "vote"])

    for cid, clip in clips.items():
        if cid in good_votes:
            vote = "good"
        elif cid in bad_votes:
            vote = "bad"
        else:
            vote = ""
        writer.writerow([cid, clip["filename"], clip["md5"],
                         clip.get("category", ""), vote])

    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        download_name="vtsearch_results.csv",
        as_attachment=True,
    )
```

### Accessing state for export

All global state is available from `vtsearch.utils`:

```python
from vtsearch.utils import clips, good_votes, bad_votes, label_history
```

| Variable         | Type                             | Contents                            |
|------------------|----------------------------------|-------------------------------------|
| `clips`          | `dict[int, dict]`                | Clip ID → clip data dict            |
| `good_votes`     | `dict[int, None]`                | Clip IDs voted "good" (ordered)     |
| `bad_votes`      | `dict[int, None]`                | Clip IDs voted "bad" (ordered)      |
| `label_history`  | `list[tuple[int, str, float]]`   | `(clip_id, "good"/"bad", timestamp)`|

Each clip dict contains at minimum: `id`, `type`, `duration`, `file_size`,
`md5`, `embedding` (numpy array), `filename`, `category`, `origin` (dict or
None), `origin_name` (str), plus media-specific fields (see
[Media Type clip data](#what-to-implement-1) below).

### Frontend integration

To make your exporter accessible from the UI, add a button or link in
`static/index.html` that calls your new endpoint.  Dataset exports typically
use `window.location.href = "/api/..."` for file downloads, while JSON exports
use `fetch()`.

### Dependencies

If your exporter needs additional packages (e.g. `openpyxl` for Excel export),
add them to `requirements.txt` (the core file) or create a dedicated
requirements file and reference it.  There is no separate aggregator file for
exporters like there is for importers — this may change if exporters get their
own plugin system in the future.

---

## Adding a Media Type

Media types define how VTSearch handles a particular kind of content: how to
embed files into vectors, how to serve clips over HTTP, what file extensions to
scan for, and what demo datasets are available.

Unlike importers, media types use **explicit registration** — you import your
class and call `register()` in `vtsearch/media/__init__.py`.

### File structure

Create a new subdirectory under `vtsearch/media/`:

```
vtsearch/media/<your_type>/
├── __init__.py       # Can be empty
├── media_type.py     # Your MediaType subclass (required)
└── requirements.txt  # Pip dependencies (required, even if empty)
```

### What to implement

Subclass `MediaType` from `vtsearch.media.base` and implement all abstract
properties and methods.

```python
# vtsearch/media/code/media_type.py

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Response, jsonify

from vtsearch.media.base import DemoDataset, MediaType


class CodeMediaType(MediaType):
    """Source code files, embedded with a code-specific model."""

    def __init__(self) -> None:
        self._model = None

    # ------------------------------------------------------------------
    # Identity (required properties)
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        """Unique internal identifier stored in each clip's 'type' field."""
        return "code"

    @property
    def name(self) -> str:
        """Human-readable label for the UI."""
        return "Source Code"

    @property
    def icon(self) -> str:
        """Emoji shown in the UI."""
        return "💻"

    # ------------------------------------------------------------------
    # File import (required properties)
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        """Glob patterns used by the folder importer to find files."""
        return ["*.py", "*.js", "*.ts", "*.go", "*.rs"]

    @property
    def folder_import_name(self) -> str:
        """Alias for the /api/dataset/load-folder endpoint.

        Defaults to type_id if not overridden.  Override to use a
        different name (e.g. audio uses "sounds" instead of "audio").
        """
        return "code"

    # ------------------------------------------------------------------
    # Viewer behaviour (required property)
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        """Whether the frontend viewer should loop.  True for audio/video."""
        return False

    # ------------------------------------------------------------------
    # Demo datasets (required property)
    # ------------------------------------------------------------------

    @property
    def demo_datasets(self) -> list:
        """Return an empty list if no demos are available yet."""
        return []
        # To add demos later:
        # return [
        #     DemoDataset(
        #         id="python_snippets",
        #         label="Python Snippets",
        #         description="Small Python functions from open-source projects.",
        #         categories=["web", "cli", "data"],
        #         source="",  # optional identifier for the raw data source
        #     ),
        # ]

    # ------------------------------------------------------------------
    # Embeddings (required methods)
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load embedding model(s) into memory.

        Called once at startup (or lazily on first use in --local mode).
        Must be idempotent — a second call should be a fast no-op.
        """
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("microsoft/codebert-base")

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for the file.

        Returns None if embedding fails (corrupt file, model not loaded, etc.).
        The vector dimensionality must be consistent across all files of this
        type AND must match the dimensionality of embed_text().
        """
        if self._model is None:
            self.load_models()
        try:
            text = file_path.read_text(errors="replace")[:8000]
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec
        except Exception:
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed a text query into the SAME vector space as embed_media().

        Used for text-query sorting: cosine similarity is computed between
        this vector and each clip's embedding.
        """
        if self._model is None:
            self.load_models()
        try:
            return self._model.encode(text, normalize_embeddings=True)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Clip data (required method)
    # ------------------------------------------------------------------

    def load_clip_data(self, file_path: Path) -> dict:
        """Return media-specific fields to merge into the clip dict.

        The base clip dict already contains: id, type, file_size, md5,
        embedding, filename, category.  You MUST include a "duration" key
        (use 0 for non-temporal media).
        """
        content = file_path.read_text(errors="replace")
        return {
            "clip_string": content,
            "duration": 0,
            "line_count": content.count("\n") + 1,
        }

    # ------------------------------------------------------------------
    # HTTP serving (required method)
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> Response:
        """Return a Flask Response serving this clip's content.

        Use flask.send_file() for binary media, flask.jsonify() for
        text/structured data.
        """
        return jsonify({
            "content": clip.get("clip_string", ""),
            "line_count": clip.get("line_count", 0),
        })
```

### Register the new type

Add two lines to `vtsearch/media/__init__.py`, alongside the existing
registrations at the bottom of the file:

```python
from vtsearch.media.code.media_type import CodeMediaType   # noqa: E402
register(CodeMediaType())
```

### What happens automatically after registration

Once registered, the rest of the application picks up your type with zero
additional changes:

| Subsystem              | What happens                                                  |
|------------------------|---------------------------------------------------------------|
| **Model init**         | `load_models()` is called at startup for your type            |
| **Folder import**      | Files matching your `file_extensions` are found and embedded  |
| **Generic media route**| `GET /api/clips/<id>/media` delegates to your `clip_response()`|
| **Text sorting**       | `embed_text()` is called for text-query cosine similarity     |
| **Demo listing**       | Your `demo_datasets` appear in `GET /api/dataset/demo-list`   |
| **Dataset export**     | Clip data is serialized to pickle (including your custom fields)|

### Abstract interface reference

**Required properties:**

| Property          | Returns     | Example                              |
|-------------------|-------------|--------------------------------------|
| `type_id`         | `str`       | `"audio"`, `"image"`, `"code"`       |
| `name`            | `str`       | `"Audio"`, `"Source Code"`           |
| `icon`            | `str`       | `"🔊"`, `"💻"`                       |
| `file_extensions` | `list[str]` | `["*.wav", "*.mp3"]`                 |
| `loops`           | `bool`      | `True` for audio/video, else `False` |
| `demo_datasets`   | `list[DemoDataset]` | See example above              |

**Optional property:**

| Property             | Returns | Default    | Purpose                          |
|----------------------|---------|------------|----------------------------------|
| `folder_import_name` | `str`   | `type_id`  | Legacy alias for folder imports  |

**Required methods:**

| Method                              | Signature                                          |
|-------------------------------------|----------------------------------------------------|
| `load_models()`                     | `() -> None`                                       |
| `embed_media(file_path)`            | `(Path) -> Optional[np.ndarray]`                   |
| `embed_text(text)`                  | `(str) -> Optional[np.ndarray]`                    |
| `load_clip_data(file_path)`         | `(Path) -> dict` (must include `"duration"` key)   |
| `clip_response(clip)`               | `(dict) -> flask.Response`                          |

### Making dataset export aware of custom clip fields

The existing `export_dataset_to_file()` in `vtsearch/datasets/loader.py`
serializes a fixed set of keys (`clip_bytes`, `clip_string`, `word_count`,
`character_count`, `width`, `height`).  If your
media type stores clip data under different keys, add those keys to the export
function so they survive a round-trip through pickle export/import.

### Frontend integration

The generic `GET /api/clips/<id>/media` endpoint works for all media types.
However, if your media type needs a specialized viewer (code highlighting,
3D rendering, etc.), you will need to add rendering logic in
`static/index.html`.  Check the clip's `type` field in the frontend JavaScript
and render accordingly.

---

## Dependency Management

VTSearch uses a layered requirements file structure to keep dependencies
modular:

```
requirements.txt              # Core deps + includes per-media + per-importer
├── vtsearch/media/audio/requirements.txt
├── vtsearch/media/video/requirements.txt
├── vtsearch/media/image/requirements.txt
├── vtsearch/media/text/requirements.txt
├── requirements-importers.txt          # Aggregates all importer deps
│   ├── vtsearch/datasets/importers/pickle/requirements.txt
│   ├── vtsearch/datasets/importers/folder/requirements.txt
│   └── vtsearch/datasets/importers/http_zip/requirements.txt
requirements-cpu.txt          # CPU-specific pins (lists packages INLINE)
requirements-gpu.txt          # GPU-specific (minimal, includes importers)
requirements-dev.txt          # Dev tools (pytest)
```

### For a new media type

1. **Create** `vtsearch/media/<type>/requirements.txt` listing any packages
   your embedder needs beyond core deps.  If none, add a comment explaining why
   it's empty:
   ```
   # Code media type — no additional dependencies beyond sentence-transformers.
   ```
2. **Add** a `-r` line to `requirements.txt`:
   ```
   -r vtsearch/media/<type>/requirements.txt
   ```
3. **Add** the packages inline to `requirements-cpu.txt` under a comment header
   (this file uses inline pins for CPU version compatibility instead of `-r`
   includes):
   ```
   # Code (vtsearch/media/code/requirements.txt)
   some-package>=1.0
   ```

### For a new importer

1. **Create** `vtsearch/datasets/importers/<name>/requirements.txt`.  Even if
   your importer has no extra deps, create the file with a comment:
   ```
   # S3 importer dependencies
   boto3
   ```
2. **Add** a `-r` line to `requirements-importers.txt`:
   ```
   -r vtsearch/datasets/importers/<name>/requirements.txt
   ```
3. **Add** packages inline to `requirements-cpu.txt` if CPU-specific pins are
   needed.

### For a new exporter

There is no dedicated aggregator file for exporters.  Add any new dependencies
directly to `requirements.txt` (or create a standalone requirements file and
reference it from `requirements.txt` with `-r`).

### Why the layered structure?

- Each component owns its own `requirements.txt` so it's obvious which packages
  belong to which feature.
- The aggregator files (`requirements.txt`, `requirements-importers.txt`) tie
  everything together for `pip install -r requirements.txt`.
- `requirements-cpu.txt` duplicates packages inline with version pins because
  CPU-only PyTorch wheels require a special `--extra-index-url` and certain
  packages need pinned versions for compatibility.
- Failed imports of an importer's sub-package emit a warning rather than
  crashing, so missing optional dependencies degrade gracefully.

---

## Quick Reference: Checklist for Each Extension Type

### New Importer Checklist

- [ ] Create `vtsearch/datasets/importers/<name>/__init__.py`
- [ ] Subclass `DatasetImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values, clips)` — populate `clips` in-place
- [ ] Expose `IMPORTER = YourImporter()` at module level
- [ ] Create `vtsearch/datasets/importers/<name>/requirements.txt`
- [ ] Add `-r` line to `requirements-importers.txt`
- [ ] Add inline deps to `requirements-cpu.txt` if needed
- [ ] Test: start the app and check `GET /api/dataset/importers` includes your importer

### New Exporter Checklist

- [ ] Add a route function to the appropriate blueprint (`datasets_bp` or `sorting_bp`)
- [ ] Import state from `vtsearch.utils` (`clips`, `good_votes`, `bad_votes`, etc.)
- [ ] Return data via `send_file()` (binary downloads) or `jsonify()` (JSON)
- [ ] Add any new dependencies to `requirements.txt`
- [ ] Add a UI trigger in `static/index.html` if needed
- [ ] Test: start the app, load a dataset, and call your endpoint

### New Media Type Checklist

- [ ] Create `vtsearch/media/<type>/` directory with `__init__.py`, `media_type.py`, `requirements.txt`
- [ ] Subclass `MediaType` and implement all abstract properties and methods
- [ ] Register in `vtsearch/media/__init__.py` with `register(YourType())`
- [ ] Add `-r` line to `requirements.txt` pointing to your `requirements.txt`
- [ ] Add inline deps to `requirements-cpu.txt`
- [ ] Update `export_dataset_to_file()` in `vtsearch/datasets/loader.py` if you use custom clip keys
- [ ] Add rendering logic in `static/index.html` if the generic viewer isn't sufficient
- [ ] Test: start the app, import a folder of your media type, verify clips appear and are sortable
