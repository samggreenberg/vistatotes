# Extending VTSearch

This guide explains how to add a new **Data Importer**, **Results Exporter**,
**Label Importer**, **Processor Importer**, or **Media Type** to VTSearch.
Each section describes the interface contract, where files go, and how to wire
up dependencies.

All four plugin systems (data importers, results exporters, label importers,
processor importers) share the same architecture: an abstract base class, a
field dataclass for UI forms, auto-discovery via `pkgutil`, and CLI support
auto-derived from field definitions.

---

## Table of Contents

1. [Adding a Data Importer](#adding-a-data-importer)
2. [Adding a Results Exporter](#adding-a-results-exporter)
3. [Adding a Label Importer](#adding-a-label-importer)
4. [Adding a Processor Importer](#adding-a-processor-importer)
5. [Adding a Media Type](#adding-a-media-type)
6. [Dependency Management (requirements.txt)](#dependency-management)

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
        from vtsearch.config import DATA_DIR
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

Results exporters deliver autodetect results to a destination (file, webhook,
email, etc.).  Like data importers, they use auto-discovery — no changes to
routes or core code are needed.

### How discovery works

The registry in `vtsearch/exporters/__init__.py` uses `pkgutil` to scan for
**sub-packages** under `vtsearch/exporters/`.  For each sub-package it finds,
it imports the module and looks for a module-level attribute named `EXPORTER`.
If that attribute exists and is a `LabelsetExporter` instance, it is registered
automatically.

Failed imports emit a warning but do not break the application.

### File structure

Create a new sub-package directory:

```
vtsearch/exporters/<your_exporter>/
├── __init__.py       # Exporter class + EXPORTER instance (required)
└── requirements.txt  # Pip dependencies, even if empty (required)
```

### What to implement

Subclass `LabelsetExporter` from `vtsearch.exporters.base` and set the
required class attributes.  Then implement the `export()` method and expose a
module-level `EXPORTER` instance.

```python
# vtsearch/exporters/sftp/__init__.py

from vtsearch.exporters.base import LabelsetExporter, ExporterField


class SftpLabelsetExporter(LabelsetExporter):
    name = "sftp"
    display_name = "SFTP Upload"
    description = "Upload results JSON to a remote SFTP server."
    icon = "📡"
    fields = [
        ExporterField("host", "Hostname", "text"),
        ExporterField("user", "Username", "text"),
        ExporterField("password", "Password", "password"),
        ExporterField(
            "path", "Remote Path", "text",
            default="/results/autodetect.json",
        ),
    ]

    def export(self, results: dict, field_values: dict) -> dict:
        """Export results to an SFTP server.

        Args:
            results: The full auto-detect results dict.  Shape:
                {
                    "media_type": "audio",
                    "detectors_run": 2,
                    "results": {
                        "detector_name": {
                            "detector_name": "...",
                            "threshold": 0.5,
                            "total_hits": 15,
                            "hits": [{...}, ...]
                        }
                    }
                }
            field_values: Mapping of ExporterField.key to user-supplied value.

        Returns:
            A dict with a "message" key (shown as confirmation to the user).
        """
        import json
        import paramiko

        host = field_values["host"]
        path = field_values["path"]

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=field_values["user"],
                    password=field_values["password"])
        sftp = ssh.open_sftp()
        with sftp.open(path, "w") as f:
            f.write(json.dumps(results, indent=2))
        sftp.close()
        ssh.close()

        return {"message": f"Uploaded to {host}:{path}"}


EXPORTER = SftpLabelsetExporter()
```

### Class attributes reference

| Attribute      | Type                  | Required | Description                                    |
|----------------|-----------------------|----------|------------------------------------------------|
| `name`         | `str`                 | Yes      | Snake_case identifier, used in API URL path    |
| `display_name` | `str`                 | Yes      | Human-readable label for the UI                |
| `description`  | `str`                 | Yes      | One-sentence subtitle                          |
| `icon`         | `str`                 | No       | Emoji/icon string (default `"📤"`)             |
| `fields`       | `list[ExporterField]` | Yes      | Ordered list of user-facing input fields       |

### ExporterField options

| Parameter    | Type        | Default  | Description                                             |
|--------------|-------------|----------|---------------------------------------------------------|
| `key`        | `str`       | —        | Field identifier (used as dict key in `field_values`)   |
| `label`      | `str`       | —        | Display label in the UI                                 |
| `field_type` | `FieldType` | —        | `"text"`, `"password"`, `"email"`, `"file"`, `"folder"`, or `"select"` |
| `description`| `str`       | `""`     | Helper text shown below the field                       |
| `options`    | `list[str]` | `[]`     | For `"select"` fields: allowed dropdown values          |
| `default`    | `str`       | `""`     | Pre-filled value                                        |
| `required`   | `bool`      | `True`   | Whether the field must be filled before exporting       |
| `placeholder`| `str`       | `""`     | Hint shown as placeholder text in the input widget      |

### How it gets invoked

1. The frontend calls `GET /api/exporters` to discover available exporters.
   The response includes your exporter's `name`, `display_name`, `description`,
   `icon`, and `fields`.
2. The user fills out the form and submits it.  The frontend sends
   `POST /api/exporters/export/<name>` with a JSON body.
3. The route handler in `vtsearch/routes/exporters.py` extracts `field_values`
   and calls `exporter.export()`.

### CLI usage

Exporters are also usable from the command line via the `--exporter` flag on
the autodetect workflow:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter sftp --host example.com --user admin --password secret --path /results.json
```

CLI arguments are auto-generated from the exporter's `fields` list.

### Wiring up dependencies

1. Create `vtsearch/exporters/<name>/requirements.txt` listing any pip
   packages your exporter needs (create the file even if empty).
2. Add a reference line to `requirements-exporters.txt`:
   ```
   -r vtsearch/exporters/<name>/requirements.txt
   ```
3. If using `requirements-cpu.txt`, add the packages inline in that file too.

### Built-in export endpoints

In addition to the exporter plugin system, VTSearch has several built-in
export endpoints:

| Endpoint                  | Method | What it exports                              | Format          |
|---------------------------|--------|----------------------------------------------|-----------------|
| `/api/dataset/export`     | GET    | Full dataset (clips + embeddings + media)    | Pickle (`.pkl`) |
| `/api/labels/export`      | GET    | LabelSet — labels with per-element origin    | JSON            |
| `/api/detector/export`    | POST   | Trained MLP weights + threshold              | JSON            |

---

## Adding a Label Importer

Label importers let users import pre-existing labels (good/bad votes) from
external sources (JSON files, CSV files, databases, etc.).  Like data
importers, they are auto-discovered at runtime.

### How discovery works

The registry in `vtsearch/labels/importers/__init__.py` scans for sub-packages
with a module-level `LABEL_IMPORTER` attribute.

### File structure

```
vtsearch/labels/importers/<your_importer>/
├── __init__.py       # Importer class + LABEL_IMPORTER instance (required)
└── requirements.txt  # Pip dependencies, even if empty (required)
```

### What to implement

Subclass `LabelImporter` from `vtsearch.labels.importers.base`.  The `run()`
method must return a list of label dicts:

```python
# vtsearch/labels/importers/postgres/__init__.py

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


class PostgresLabelImporter(LabelImporter):
    name = "postgres"
    display_name = "PostgreSQL Query"
    description = "Import labels from a PostgreSQL database query."
    icon = "🐘"
    fields = [
        LabelImporterField("host", "Hostname", "text"),
        LabelImporterField("database", "Database", "text"),
        LabelImporterField(
            "query", "SQL Query", "text",
            description="Must return md5 and label columns.",
        ),
    ]

    def run(self, field_values: dict) -> list[dict]:
        """Return a list of label dicts.

        Each dict must have "md5" and "label" keys.  Labels must be
        "good" or "bad"; any other value is skipped by the route handler.
        """
        import psycopg2

        conn = psycopg2.connect(
            host=field_values["host"],
            database=field_values["database"],
        )
        cur = conn.cursor()
        cur.execute(field_values["query"])
        return [{"md5": row[0], "label": row[1]} for row in cur.fetchall()]


LABEL_IMPORTER = PostgresLabelImporter()
```

### How it gets invoked

1. `GET /api/label-importers` returns the list of available label importers.
2. `POST /api/label-importers/import/<name>` invokes `run()` and applies the
   returned labels to the current dataset by matching clip MD5 hashes.

---

## Adding a Processor Importer

Processor importers let users import processors (detectors/extractors) from
external sources.  A processor importer takes some input (a JSON detector
file, a labeled media file, etc.) and returns a dict containing model weights
and a threshold — which is then saved as a favorite detector.

### How discovery works

The registry in `vtsearch/processors/importers/__init__.py` scans for
sub-packages with a module-level `PROCESSOR_IMPORTER` attribute.

### File structure

```
vtsearch/processors/importers/<your_importer>/
├── __init__.py       # Importer class + PROCESSOR_IMPORTER instance (required)
└── requirements.txt  # Pip dependencies, even if empty (required)
```

### What to implement

Subclass `ProcessorImporter` from `vtsearch.processors.importers.base`.  The
`run()` method must return a dict with `media_type`, `weights`, and
`threshold` keys:

```python
# vtsearch/processors/importers/s3/__init__.py

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class S3ProcessorImporter(ProcessorImporter):
    name = "s3"
    display_name = "S3 Detector File"
    description = "Download a detector JSON file from an S3 bucket."
    icon = "☁️"
    fields = [
        ProcessorImporterField("bucket", "S3 Bucket", "text"),
        ProcessorImporterField("key", "Object Key", "text"),
    ]

    def run(self, field_values: dict) -> dict:
        """Download and parse a detector JSON from S3.

        Must return a dict with at minimum:
            - "media_type" (str): e.g. "audio", "image"
            - "weights" (dict): MLP state dict as nested lists
            - "threshold" (float): decision boundary in [0, 1]
        """
        import json
        import boto3

        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=field_values["bucket"], Key=field_values["key"])
        data = json.loads(obj["Body"].read())
        return {
            "media_type": data.get("media_type", "audio"),
            "weights": data["weights"],
            "threshold": data.get("threshold", 0.5),
        }


PROCESSOR_IMPORTER = S3ProcessorImporter()
```

### How it gets invoked

1. `GET /api/processor-importers` returns the list of available importers.
2. `POST /api/processor-importers/import/<name>` invokes `run()`, combines the
   result with the user-supplied name, and saves it as a favorite detector.

### CLI usage

Processor importers are used from the CLI via the settings file.  Add a
processor recipe to `favorite_processors` in `settings.json`:

```json
{
    "favorite_processors": [
        {
            "processor_name": "my detector",
            "processor_importer": "s3",
            "field_values": {"bucket": "my-bucket", "key": "detector.json"}
        }
    ]
}
```

Then run autodetect:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json
```

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

from vtsearch.media.base import DemoDataset, MediaResponse, MediaType


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

    def clip_response(self, clip: dict) -> MediaResponse:
        """Return a MediaResponse serving this clip's content.

        Set data to bytes for binary media (with an appropriate mimetype)
        or to a dict for JSON responses.
        """
        return MediaResponse(
            data={
                "content": clip.get("clip_string", ""),
                "line_count": clip.get("line_count", 0),
            },
            mimetype="application/json",
        )
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
| `clip_response(clip)`               | `(dict) -> MediaResponse`                           |

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
requirements.txt              # Core deps + includes per-media + per-importer + per-exporter
├── vtsearch/media/audio/requirements.txt
├── vtsearch/media/video/requirements.txt
├── vtsearch/media/image/requirements.txt
├── vtsearch/media/text/requirements.txt
├── requirements-importers.txt          # Aggregates all data importer deps
│   ├── vtsearch/datasets/importers/pickle/requirements.txt
│   ├── vtsearch/datasets/importers/folder/requirements.txt
│   └── vtsearch/datasets/importers/http_zip/requirements.txt
├── requirements-exporters.txt          # Aggregates all exporter deps
│   ├── vtsearch/exporters/gui/requirements.txt
│   ├── vtsearch/exporters/file/requirements.txt
│   └── vtsearch/exporters/email_smtp/requirements.txt
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

1. **Create** `vtsearch/exporters/<name>/requirements.txt`.  Even if your
   exporter has no extra deps, create the file with a comment.
2. **Add** a `-r` line to `requirements-exporters.txt`:
   ```
   -r vtsearch/exporters/<name>/requirements.txt
   ```
3. **Add** packages inline to `requirements-cpu.txt` if CPU-specific pins are
   needed.

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

- [ ] Create `vtsearch/exporters/<name>/__init__.py`
- [ ] Subclass `LabelsetExporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `export(self, results, field_values)` — return a dict with a `"message"` key
- [ ] Expose `EXPORTER = YourExporter()` at module level
- [ ] Create `vtsearch/exporters/<name>/requirements.txt`
- [ ] Add `-r` line to `requirements-exporters.txt`
- [ ] Add inline deps to `requirements-cpu.txt` if needed
- [ ] Test: start the app and check `GET /api/exporters` includes your exporter

### New Label Importer Checklist

- [ ] Create `vtsearch/labels/importers/<name>/__init__.py`
- [ ] Subclass `LabelImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a list of `{"md5": ..., "label": ...}` dicts
- [ ] Expose `LABEL_IMPORTER = YourImporter()` at module level
- [ ] Create `vtsearch/labels/importers/<name>/requirements.txt`
- [ ] Test: start the app and check `GET /api/label-importers` includes your importer

### New Processor Importer Checklist

- [ ] Create `vtsearch/processors/importers/<name>/__init__.py`
- [ ] Subclass `ProcessorImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a dict with `media_type`, `weights`, `threshold`
- [ ] Expose `PROCESSOR_IMPORTER = YourImporter()` at module level
- [ ] Create `vtsearch/processors/importers/<name>/requirements.txt`
- [ ] Test: start the app and check `GET /api/processor-importers` includes your importer

### New Media Type Checklist

- [ ] Create `vtsearch/media/<type>/` directory with `__init__.py`, `media_type.py`, `requirements.txt`
- [ ] Subclass `MediaType` and implement all abstract properties and methods
- [ ] Register in `vtsearch/media/__init__.py` with `register(YourType())`
- [ ] Add `-r` line to `requirements.txt` pointing to your `requirements.txt`
- [ ] Add inline deps to `requirements-cpu.txt`
- [ ] Update `export_dataset_to_file()` in `vtsearch/datasets/loader.py` if you use custom clip keys
- [ ] Add rendering logic in `static/index.html` if the generic viewer isn't sufficient
- [ ] Test: start the app, import a folder of your media type, verify clips appear and are sortable
