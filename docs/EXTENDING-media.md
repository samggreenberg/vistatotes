# Extending VTSearch: Media System

How to add new content types, embedders, clippers, converters, and media
sources. Media plugins are auto-discovered via sentinel attributes on
sub-packages of `vtscore/media/`.

**Related docs:** [EXTENDING.md](EXTENDING.md) (index, checklists, auth,
dependencies) · [EXTENDING-plugins.md](EXTENDING-plugins.md) (importers,
exporters, sources for datasets/labels/settings) ·
[EXTENDING-processors.md](EXTENDING-processors.md) (detectors, localizers,
extractors).

## Contents

- [Media System](#media-system): discovery, sentinels, registration
- [Adding a Media Type](#adding-a-media-type): e.g. a new content kind
- [Adding a Media Embedder](#adding-a-media-embedder): alternative or new
  embedding model
- [Adding a Media Clipper](#adding-a-media-clipper): cut clips out of a
  longer source file
- [Adding a Media Cleaner](#adding-a-media-cleaner): strip content-free
  regions from each item before it is embedded
- [Adding a Media Converter](#adding-a-media-converter): transform
  between media types (e.g. document to image)
- [Adding a Media Source](#adding-a-media-source): resolve media bytes
  from an origin (local, HTTP archive, custom backend)

---

## Media System

Media types, embedders, clippers, and cleaners are **auto-discovered** at
import time. The `_discover_media_plugins()` function in
`vtscore/media/__init__.py` scans sub-packages of `vtscore/media/` for
module-level sentinel attributes:

| Sentinel     | Location                         | Type                 | Description                          |
|--------------|----------------------------------|----------------------|--------------------------------------|
| `MEDIA_TYPE` | media-type package `__init__.py` | `MediaType`          | A single media type instance         |
| `CLIPPERS`   | media-type package `__init__.py` | `list[MediaClipper]` | Clipper instances (may be empty)     |
| `CLEANERS`   | media-type package `__init__.py` | `list[MediaCleaner]` | Cleanup-gate instances (may be empty) |
| `EMBEDDER`   | an `embedder_<name>.py` file inside the media-type package | `MediaEmbedder` | One embedder per module              |

Embedders use **one module per embedder**: any `embedder_<name>.py` file
inside a media-type package is auto-loaded and its module-level `EMBEDDER`
sentinel is registered. Symlinked directories **and** symlinked embedder
files are both supported, so a custom embedder can live outside the
VTSearch tree and be wired in by symlinking a single file into the
appropriate media-type package. No edits to any `__init__.py` are required.

Third-party or project-specific types can still be registered manually
via `register()`, `register_embedder()`, `register_clipper()`, and
`register_cleaner()`.

---

## Adding a Media Type

Media types define how VTSearch handles a particular kind of content: how
to serve clips over HTTP, what file extensions to scan for, what demo
datasets are available, and how to load media-specific fields from files.

**Library contract:**
[`vtscore/docs/extending/media-types.md`](../vtscore/docs/extending/media-types.md)
states the same contract from the library side, and is the guide to follow
when shipping a media type as a separate distribution rather than adding one
to this repo. This section is the in-repo path: the same contract plus the
app-tier wiring.

### File structure

```
vtscore/media/<your_type>/
├── __init__.py       # Must expose MEDIA_TYPE and CLIPPERS sentinels
├── media_type.py     # Your MediaType subclass (required)
└── embedder_<name>.py  # Optional; one file per embedder, each exposing EMBEDDER
```

### What to implement

Subclass `MediaType` from `vtscore.media.base` and implement all abstract
properties and methods.

```python
# vtscore/media/code/media_type.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtscore.media.base import DemoDataset, MediaResponse, MediaType


class CodeMediaType(MediaType):
    """Source code files."""

    # --- Identity (required abstract properties) ---

    @property
    def type_id(self) -> str:
        return "code"

    @property
    def name(self) -> str:
        return "Source Code"

    @property
    def icon(self) -> str:
        return "code"  # SVG icon type name for the UI

    # --- File import (required abstract property) ---

    @property
    def file_extensions(self) -> list:
        return ["*.py", "*.js", "*.ts", "*.go", "*.rs"]

    # --- Viewer behaviour (required abstract property) ---

    @property
    def loops(self) -> bool:
        return False

    # --- Demo datasets (required abstract property) ---

    @property
    def demo_datasets(self) -> list:
        return []  # No demos yet

    # --- Media data (required abstract method) ---

    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        """Return media-specific fields to merge into the media dict.

        The base media dict already contains: id, type, file_size, md5,
        embedding, filename, category.  You MUST include a "duration" key
        (use 0 for non-temporal media).
        """
        content = file_path.read_text(errors="replace")
        return {
            "media_string": content,
            "duration": 0,
            "line_count": content.count("\n") + 1,
        }

    # --- HTTP serving (required abstract method) ---

    def media_response(self, media: dict) -> MediaResponse:
        """Return a MediaResponse for HTTP serving.

        Use _resolve_media_bytes() for binary media or
        _resolve_media_string() for text media to support both
        preloaded and thin (lazy-loaded) modes.
        """
        content = self._resolve_media_string(media)
        return MediaResponse(
            data={"content": content, "line_count": media.get("line_count", 0)},
            mimetype="application/json",
        )
```

### Register the new type

Expose the sentinels in your sub-package's `__init__.py`:

```python
# vtscore/media/code/__init__.py

from vtscore.media.code.media_type import CodeMediaType

MEDIA_TYPE = CodeMediaType()
CLIPPERS = []  # No clippers yet (add when needed)
```

Each embedder lives in its own `embedder_<name>.py` file with an `EMBEDDER`
sentinel at the bottom:

```python
# vtscore/media/code/embedder_codebert.py

# ... class definition ...

EMBEDDER = CodeBertEmbedder()
```

The auto-discovery system finds these sentinels at import time. No
changes to `vtscore/media/__init__.py` are needed.

### MediaType abstract interface reference

**Required abstract properties:**

| Property          | Returns     | Example                              |
|-------------------|-------------|--------------------------------------|
| `type_id`         | `str`       | `"audio"`, `"image"`, `"code"`       |
| `name`            | `str`       | `"Audio"`, `"Source Code"`           |
| `icon`            | `str`       | `"audio"`, `"code"` (SVG icon type name) |
| `file_extensions` | `list[str]` | `["*.wav", "*.mp3"]`                 |
| `loops`           | `bool`      | `True` for audio/video, else `False` |
| `demo_datasets`   | `list[DemoDataset]` | See example above              |

**Required abstract methods:**

| Method                      | Signature                      | Description                              |
|-----------------------------|--------------------------------|------------------------------------------|
| `load_media_data(file_path, media_bytes=None)`| `(Path, bytes \| None) -> dict` | Must include `"duration"` key. The optional `media_bytes` lets callers (e.g. the folder loader) pass already-read bytes to avoid a second disk read. |
| `media_response(media)`     | `(dict) -> MediaResponse`      | HTTP response for a media item           |

> **Pixel-media types: report *displayed* dimensions.** A media's stored
> `width`/`height` are the coordinate space every box in the system is expressed
> against — clip boxes, extractor and localizer output, the region a user drags
> on the canvas. For images that space is the picture *as displayed*, i.e. after
> EXIF orientation: a phone photo shot sideways stores a landscape bitmap plus a
> "rotate me" tag, and browsers honour the tag, so the user is looking at a
> portrait image. `vtscore.media.image.decode` applies the tag on every decode
> (`decode_bounded`, `decode_bounded_rgb`, `open_upright`) and exposes
> `upright_size()` for the header-only read `load_media_data` wants. Reach for
> `open_image()` only when you genuinely want the untransposed original — it is
> the lazy, metadata-only escape hatch, not the default.

**Optional overridable properties (with defaults):**

| Property             | Returns     | Default            | Purpose                                   |
|----------------------|-------------|--------------------|-------------------------------------------|
| `folder_import_name` | `str`       | `type_id`          | Alias for folder imports (matches `type_id`) |
| `dir_key`            | `str`       | `type_id + "_dir"` | Key in pickle files for external dir       |
| `pickle_extra_fields`| `list[str]` | `[]`               | Extra fields to preserve in pickle round-trips (e.g. `["width", "height"]`) |
| `importable`         | `bool`      | `True`             | Whether the user imports this type natively (folder scan). Set `False` for a *convert-in* half type produced only by a converter (e.g. `face`) |
| `converts_to`        | `list[str]` | `[]`               | For a non-embeddable *convert-out* half type (e.g. `document`), the embeddable targets it converts into (first = default), e.g. `["image", "text"]` |

> **Half media types.** `MediaType` separates two orthogonal capabilities:
> `importable` (a native ingestion category) and `embeddable` (has a registered
> embedder — derived automatically). A *convert-out* half type is importable
> but not embeddable and declares `converts_to` (`document`); a *convert-in*
> half type is embeddable but sets `importable = False` and has empty
> `file_extensions` (`face`, produced by the `image2face` converter). Full
> types are both.

**Optional overridable methods:**

| Method                        | Signature                          | Description                        |
|-------------------------------|------------------------------------|------------------------------------|
| `display_metadata(media)`     | `(dict) -> dict[str, Any]`         | Metadata for the labeling UI       |
| `image_response(media)`       | `(dict) -> MediaResponse \| None`  | A *paintable image* for the media (waveform, frame, first page, crop) for every surface that shows a picture rather than plays the media. Defaults to `None` — "this type has no visual form" |
| `ensure_thumbnail_bytes(media)` | `(dict) -> bytes \| None`        | Generate + memoise `media["thumbnail_bytes"]` from the media's *resolvable* bytes, for media that had no file at ingest. Defaults to a plain read of what's cached (no generation) |
| `load_models()`               | `() -> None`                       | Load inline embedding models (legacy) |
| `embed_text(text)`            | `(str) -> Optional[np.ndarray]`    | Inline text embedding (legacy)     |
| `load_demo_source(...)`       | See docstring                      | Download and embed a demo dataset  |

> **Always merge the base `display_metadata`.** Override it to *prepend* your
> type-specific fields, then fold in whatever the base produced that you did
> not already set — the shipped types all end with
> `result.update({k: v for k, v in super().display_metadata(media).items() if k not in result})`.
> The base contributes the shared fields (Category, File Size, the `Clip *`
> provenance) plus the **"AI Caption" / "AI Tags"** row, which surfaces the
> per-media signpost text a Browse-prepped dataset already carries (see
> `vtscore/projection/signpost_texts.py`). A type that returns its own dict
> without merging silently drops all of them.

> **Override `image_response` if your type has a picture but is not one.**
> `media_response` serves your media's *own* bytes, and for most types those
> are not an image: audio is a WAV, video an MP4, a document an
> `application/pdf`. Every surface that shows a **picture** of a media — the
> grid tile, the VTSBrowse bin popup, the labeling thumbnail,
> `GET /api/medias/<id>/image` and `/api/medias/<id>/thumbnail` — calls
> `image_response` instead, and paints a placeholder when it returns `None`.
> Audio returns its waveform PNG, video its mid-frame, `document` its first
> page rasterised, `face` the crop; `text` has no visual form and keeps the
> `None` default, and so does `image`, whose source bytes the routes stream
> directly without consulting the hook. Generate through
> `ensure_thumbnail_bytes` (below) rather than inline, so the bytes you serve
> match the ones the warm-up pass produces.

> **Override `ensure_thumbnail_bytes` if your type has a browsable thumbnail.**
> The path-based hooks above only fire for media the loader can read at ingest.
> An **archive-member** media (`local_archive_member`) has no file at all — only
> `{archive path, member}` — so it leaves import with no thumbnail, and every
> browse tile would stream a tar member and decode it on the request thread.
> `ensure_thumbnail_bytes` is the type-agnostic way to build one from
> `_resolve_media_bytes(media)`; the background pass in
> `vtscore/datasets/thumbnail_warm.py` calls it per media after a load, and the
> serving path calls it too, so a warmed thumbnail is byte-identical to a lazily
> generated one. Generate through the same helper your `image_response` fallback
> uses, memoise onto `media["thumbnail_bytes"]`, and never retain the resolved
> payload on the media.

### What happens automatically after registration

| Subsystem              | What happens                                                  |
|------------------------|---------------------------------------------------------------|
| **Folder import**      | Files matching your `file_extensions` are found and embedded  |
| **Generic media route**| `GET /api/medias/<id>/media` delegates to your `media_response()`|
| **Demo listing**       | Your `demo_datasets` appear in `GET /api/dataset/demo-list`   |
| **Dataset export**     | Clip data is serialized to pickle (including custom fields)   |
| **Media types API**    | `GET /api/media-types` includes your type's metadata          |

### Making dataset export aware of custom clip fields

If your media type stores clip data under non-standard keys, override
`pickle_extra_fields` to return those key names so they survive pickle
export/import. For example:

```python
@property
def pickle_extra_fields(self) -> list[str]:
    return ["line_count"]
```

---

## Adding a Media Embedder

Media embedders produce fixed-size vector embeddings from media files and
text queries. Each embedder is associated with exactly one media type but a
media type may have multiple embedders.

**Library contract:**
[`vtscore/docs/extending/embedders.md`](../vtscore/docs/extending/embedders.md)
states the same contract from the library side, and is the guide to follow
when shipping an embedder as a separate distribution rather than adding one
to this repo. This section is the in-repo path: the same contract plus the
app-tier wiring.

### File structure

```
vtscore/media/<type>/
└── embedder_<name>.py    # One file per embedder, each exposing EMBEDDER
```

Every embedder lives in its own `embedder_<name>.py` file. Exactly one
embedder per media type should override the `is_default` property to return
`True`; that embedder is what callers using
`embedders_for_type(t)[0]` receive. Existing embedders:

<!-- BEGIN GENERATED: embedder-files -->
<!-- Generated by scripts/gen-docs-inventories.py; do not edit by hand. Refresh with: python scripts/gen-docs-inventories.py -->

| File (under `vtscore/media/`) | Class | Embedder | Media type | Default |
|---|---|---|---|---|
| `audio/embedder_clap_general.py` | `AudioClapGeneralEmbedder` | `clap_general` | `audio` | ✅ |
| `audio/embedder_ast.py` | `AudioASTEmbedder` | `ast` | `audio` |  |
| `audio/embedder_beats.py` | `AudioBEATsEmbedder` | `beats` | `audio` |  |
| `audio/embedder_clap.py` | `AudioClapEmbedder` | `clap` | `audio` |  |
| `audio/embedder_clap_music.py` | `AudioClapMusicEmbedder` | `clap_music` | `audio` |  |
| `audio/embedder_paraspeechclap.py` | `AudioParaSpeechClapEmbedder` | `paraspeechclap` | `audio` |  |
| `audio/embedder_whisper.py` | `AudioWhisperEncoderEmbedder` | `whisper_encoder` | `audio` |  |
| `face/embedder_facenet.py` | `FaceEmbedder` | `face` | `face` |  |
| `image/embedder_siglip.py` | `ImageSiglipEmbedder` | `siglip` | `image` | ✅ |
| `image/embedder_clip.py` | `ImageClipEmbedder` | `clip` | `image` |  |
| `image/embedder_clip_l.py` | `ImageClipLargeEmbedder` | `clip_l` | `image` |  |
| `image/embedder_dinov2_patch.py` | `ImageDinov2PatchEmbedder` | `dinov2_patch` | `image` |  |
| `image/embedder_dinov2_single.py` | `ImageDinov2SingleEmbedder` | `dinov2_single` | `image` |  |
| `image/embedder_dinov3_patch.py` | `ImageDinov3PatchEmbedder` | `dinov3_patch` | `image` |  |
| `image/embedder_dinov3_single.py` | `ImageDinov3SingleEmbedder` | `dinov3_single` | `image` |  |
| `image/embedder_eupe_patch.py` | `ImageEupePatchEmbedder` | `eupe_patch` | `image` |  |
| `image/embedder_eupe_single.py` | `ImageEupeSingleEmbedder` | `eupe_single` | `image` |  |
| `image/embedder_sift_vlad.py` | `ImageSiftVladEmbedder` | `sift_vlad` | `image` |  |
| `image/embedder_siglip2.py` | `ImageSiglip2Embedder` | `siglip2` | `image` |  |
| `image/embedder_siglip2_l.py` | `ImageSiglip2LEmbedder` | `siglip2_l` | `image` |  |
| `image/embedder_siglip_l.py` | `ImageSiglipLEmbedder` | `siglip_l` | `image` |  |
| `text/embedder_e5.py` | `TextE5Embedder` | `e5` | `text` | ✅ |
| `text/embedder_bge.py` | `TextBGEEmbedder` | `bge` | `text` |  |
| `video/embedder_xclip.py` | `VideoXClipEmbedder` | `xclip` | `video` | ✅ |
| `video/embedder_languagebind.py` | `VideoLanguageBindEmbedder` | `languagebind` | `video` |  |
| `video/embedder_videomae.py` | `VideoVideoMAEEmbedder` | `videomae` | `video` |  |

<!-- END GENERATED: embedder-files -->

### What to implement

Subclass `MediaEmbedder` from `vtscore.media.embedder`.

```python
# vtscore/media/code/embedder_codebert.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vtscore.media.embedder import MediaEmbedder


class CodeBertEmbedder(MediaEmbedder):
    """Embeds source code using CodeBERT."""

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    # --- Identity (required abstract properties) ---

    @property
    def name(self) -> str:
        """Unique identifier (also the registry key)."""
        return "codebert"

    @property
    def media_type_id(self) -> str:
        """The type_id of the media type this embedder works with."""
        return "code"

    @property
    def is_default(self) -> bool:
        """Mark this embedder as the default for its media type.

        Exactly one embedder per media type should override this to ``True``;
        callers using ``embedders_for_type(t)[0]`` receive that one.
        """
        return True

    # --- Model lifecycle (required abstract method) ---

    def _load_models_impl(self) -> None:
        """Load the embedding model. Must be idempotent.

        Override ``_load_models_impl`` (not ``load_models``).
        The public ``load_models()`` wrapper handles locking and
        ImportError wrapping automatically.
        """
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._on_progress("loading", "Loading CodeBERT…", 0, 0)
        self._model = SentenceTransformer("microsoft/codebert-base")

    # --- Embedding (required abstract method) ---

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for a media item.

        *media* is a media dict.  File-based embedders read
        ``Path(media["media_path"])``.  Service-based embedders can instead
        use ``media["origin"]``, ``media["origin_name"]``, or
        ``media.get("custom_metadata")`` to look up the content remotely
        (e.g. by a ``content_id`` stashed in ``origin.params``).

        Override ``_embed_media_impl`` (not ``embed_media``).  The public
        ``embed_media()`` wrapper acquires a global lock so that only one
        forward pass runs at a time.

        Returns None if embedding fails.  The vector dimensionality must
        be consistent and must match embed_text().
        """
        if self._model is None:
            self.load_models()
        try:
            text = Path(media["media_path"]).read_text(errors="replace")[:8000]
            return self._model.encode(text, normalize_embeddings=True)
        except Exception:
            return None

    # --- Optional: text embedding ---
    # Override `_embed_text_impl` (the subclass hook), NOT `embed_text`.
    # `embed_text` is a framework wrapper that L2-normalises the returned
    # vector so text queries stay unit-norm — cosine and dot-product
    # scorers downstream depend on that. A plugin that overrides
    # `embed_text` directly and forgets to normalise silently poisons
    # every score.

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        """Embed a text query into the SAME vector space as _embed_media_impl.

        Used for text-query sorting. Default returns None (no text sort).
        The framework wraps this with `embed_text`, which handles locking
        and L2-normalisation.
        """
        if self._model is None:
            self.load_models()
        try:
            return self._model.encode(text, normalize_embeddings=True)
        except Exception:
            return None
```

### Decoding audio: use `decode_audio`, never `librosa.load`

Any plugin that turns audio into samples — an embedder, a clipper, a
converter, a captioner — must go through
`vtscore.media.audio.decode.decode_audio`:

```python
from vtscore.media.audio.decode import decode_audio

samples, sr = decode_audio(source, sr=MY_SAMPLE_RATE, mono=True)
```

*source* may be a path, raw `bytes`, or a file-like object; the return is a
C-contiguous `float32` array in [-1, 1], mono-downmixed by channel mean, at
`sr` (or the native rate when `sr=None`). `offset` / `duration` (seconds)
select a window before any resampling. Failure raises `AudioDecodeError`.

Do **not** call `librosa.load`. It falls back to `audioread` for every
container `libsndfile` can't parse — which is all of AAC/M4A/MP4 — and that
fallback is removed in librosa 1.0, so those codecs would break silently.
`decode_audio` uses `soundfile` for the native formats and pipes everything
else through the bundled ffmpeg over `stdin`, with no temp-file spill.
librosa is still the right tool for *analysis* (`librosa.effects.split`,
`librosa.feature.melspectrogram`, `librosa.cqt`); it is only `load` that is
off-limits.

### Service-based embedders

Embedders are not required to read a local file.  Because `_embed_media_impl`
receives the full media dict, a service-based embedder can resolve content
remotely from whatever identifier its importer stashed in
`origin["params"]` or `media["custom_metadata"]`:

```python
def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
    content_id = (media.get("origin") or {}).get("params", {}).get("content_id")
    if not content_id:
        return None  # no server identifier → cannot embed
    return self._client.get_embedding(content_id)
```

For services that natively accept many items per request, override
`_embed_media_bulk_impl(medias)` to send one request (or do internal
batching). The loader always calls `embed_media_bulk` with every
pending file; the default implementation loops over `embed_media`
per item and emits progress via `self._on_progress`, so custom
overrides that batch internally should emit their own progress too.

Importer code that has already built a `dict[int, dict]` of medias
(keyed by media ID) can call the dict-shaped sugar wrapper directly:

```python
# medias: dict[int, dict] keyed by 1-based media ID
from vtscore.embedding.media_vectors import set_media_embedding

vectors = emb.embed_medias(medias)  # -> dict[int, Optional[np.ndarray]]
for media_id, vec in vectors.items():
    if vec is not None:
        set_media_embedding(medias[media_id], emb.name, vec)
```

`embed_medias` delegates to `embed_media_bulk` internally. Subclasses
only need to override `_embed_media_bulk_impl` to gain a native bulk
path; the dict wrapper picks it up automatically.

### Register the embedder

Drop the embedder into an `embedder_<name>.py` file inside the media-type
package and expose an `EMBEDDER` sentinel at the bottom. Discovery is
automatic; no edits to `__init__.py` are needed:

```python
# vtscore/media/code/embedder_codebert.py

class CodeBertEmbedder(MediaEmbedder):
    ...

EMBEDDER = CodeBertEmbedder()
```

For an alternative embedder on an **existing** media type, drop a new
`embedder_<name>.py` file into that type's package (e.g.
`vtscore/media/image/embedder_myclip.py`) with its own `EMBEDDER`
sentinel. To wire in a custom embedder living outside the VTSearch
source tree, symlink the file in. Symlinked embedder modules are
loaded via `spec_from_file_location` so discovery still works.

### MediaEmbedder abstract interface reference

**Required abstract properties:**

| Property        | Returns | Description                              |
|-----------------|---------|------------------------------------------|
| `name`          | `str`   | Unique identifier (e.g. `"clap"`, `"clip"`) |
| `media_type_id` | `str`  | Which media type this embedder works with |

**Required abstract methods:**

| Method                      | Signature                              | Description                    |
|-----------------------------|----------------------------------------|--------------------------------|
| `_load_models_impl()`       | `() -> None`                           | Load model; must be idempotent. Override this, not `load_models()` |
| `_embed_media_impl(media)`  | `(dict) -> Optional[np.ndarray]`       | Embed a single media item. Override this, not `embed_media()`      |

**Optional overridable methods:**

| Method                                | Signature                                          | Description                          |
|---------------------------------------|----------------------------------------------------|--------------------------------------|
| `_embed_text_impl(text)`              | `(str) -> Optional[np.ndarray]`                    | Embed a text query (default: `None`). Override this, not `embed_text` — the public wrapper handles locking and L2-normalisation. |
| `embed_text_enriched(text)`           | `(str) -> Optional[np.ndarray]`                    | Average over `description_wrappers`  |
| `_embed_media_bulk_impl(medias)`      | `(list[dict]) -> list[Optional[np.ndarray]]`       | Embed a list of medias. Default loops over `embed_media` with per-item progress. Override for a native bulk path (e.g. a remote API that accepts many items per request); overrides that batch internally must emit their own progress through `self._on_progress`. |
| `models_loaded()`                     | `() -> bool`                                       | Whether the model is already resident in this process, without loading it. Default reads the same private model attribute `load_models()` sets; override it alongside `loaded_backbone()` if the backbone lives elsewhere. Read by code that plans around the load rather than performing it — the text-sort bar budgets nothing for its model-load step when this is `True`. |

**Convenience wrappers (don't override these):**

| Method                                | Signature                                                       | Description                          |
|---------------------------------------|-----------------------------------------------------------------|--------------------------------------|
| `embed_media(media)`                  | `(dict) -> Optional[np.ndarray]`                                | Single-item public entrypoint. Acquires `_embed_lock`, calls `_embed_media_impl`, L2-normalises the result. |
| `embed_text(text)`                    | `(str) -> Optional[np.ndarray]`                                 | Single-item text-query entrypoint. Calls `_embed_text_impl` and L2-normalises the result. |
| `embed_media_bulk(medias)`            | `(list[dict]) -> list[Optional[np.ndarray]]`                    | List-shaped public entrypoint. Calls `_embed_media_bulk_impl` and short-circuits empty input. |
| `embed_medias(medias)`                | `(dict[int, dict]) -> dict[int, Optional[np.ndarray]]`          | Sugar for callers with id-keyed medias (e.g. importers). Delegates to `embed_media_bulk`, pairs vectors back to input keys. |

**Optional overridable properties:**

| Property               | Returns     | Description                                |
|------------------------|-------------|--------------------------------------------|
| `description_wrappers` | `list[str]` | Templates with `{text}` for enriched embedding (e.g. `["the sound of {text}"]`). Default `[]` — see below |

Whether a prompt ensemble helps is a property of the **embedder**, not of the
media type, and the default (`[]`) is a real answer rather than an unfilled
slot: issue #3127 measured enrichment on/off over 22 eval datasets and 560
paired queries and found it a clear loss on `e5`, `bge`, `siglip` and `clap`
(text enrichment lost on 45 of 45 categories), and a gain only on
`clap_general` and `xclip`.  Issue #3341 therefore emptied the list on the four
losers, so the *Enrich descriptions* setting is a no-op there instead of a
small, silent cost.  Leave `description_wrappers` empty unless you have
measured a gain on **your** checkpoint; a sibling model's templates are not
evidence.

**Instance attributes:**

| Attribute       | Type               | Description                         |
|-----------------|--------------------|-------------------------------------|
| `_on_progress`  | `ProgressCallback` | Progress callback (default: no-op). Process-wide default set via `set_progress_callback()`; **reads and writes are per-thread**, so wrap a temporary redirect in `with emb.progress_scope(cb):` rather than saving/restoring by hand. Embedders are singletons — a process-wide swap would cross concurrent loads' progress bars and cancellations. A background warm-up with no progress surface should use `with emb.silent_progress():`; an unscoped `load_models()` reports on the process-wide default and parks it `idle` when it returns. |

### Built-in embedders

| Embedder | Name | Media Type | Model | Dimensions |
|----------|------|------------|-------|------------|
| `AudioClapEmbedder` | `clap` | `audio` | LAION CLAP (laion/clap-htsat-unfused) | 512 |
| `AudioClapMusicEmbedder` | `clap_music` | `audio` | CLAP Music & Speech (laion/larger_clap_music_and_speech) | 512 |
| `AudioClapGeneralEmbedder` | `clap_general` | `audio` | CLAP General 2024 (laion/larger_clap_general) | 512 |
| `AudioParaSpeechClapEmbedder` | `paraspeechclap` | `audio` | ParaSpeechCLAP speech-style (WavLM-Large + Granite, ajd12342/paraspeechclap-combined) | 768 |
| `AudioBEATsEmbedder` | `beats` | `audio` | BEATs iter3+ AS2M self-supervised encoder (lpepino/beats_ckpts), audio-only | 768 |
| `AudioASTEmbedder` | `ast` | `audio` | AST audio spectrogram (MIT/ast-finetuned-audioset-10-10-0.4593), audio-only | 768 |
| `AudioWhisperEncoderEmbedder` | `whisper_encoder` | `audio` | Whisper-base encoder (openai/whisper-base), audio-only | 512 |
| `ImageSiglipEmbedder` | `siglip` | `image` | SigLIP (google/siglip-base-patch16-224) | 768 |
| `ImageSiglip2Embedder` | `siglip2` | `image` | SigLIP 2 (google/siglip2-base-patch16-224) | 768 |
| `ImageSiglip2LEmbedder` | `siglip2_l` | `image` | SigLIP2-L (google/siglip2-so400m-patch14-384) | 1152 |
| `ImageClipEmbedder` | `clip` | `image` | CLIP (openai/clip-vit-base-patch32) | 512 |
| `ImageDinov2SingleEmbedder` / `ImageDinov2PatchEmbedder` | `dinov2_single` / `dinov2_patch` | `image` | DINOv2 ViT-B/14 (facebook/dinov2-base), ungated | 768 |
| `ImageDinov3SingleEmbedder` / `ImageDinov3PatchEmbedder` | `dinov3_single` / `dinov3_patch` | `image` | DINOv3 ViT-B/16 (facebook/dinov3-vitb16-pretrain-lvd1689m), HF-gated | 768 |
| `ImageEupeSingleEmbedder` / `ImageEupePatchEmbedder` | `eupe_single` / `eupe_patch` | `image` | EUPE ViT-B/16 (facebookresearch/EUPE), FAIR Noncommercial Research Licence | 768 |
| `ImageSiftVladEmbedder` | `sift_vlad` | `image` | SIFT/VLAD instance matching (classical, no text encoder) | 8192 (64 × 128) |
| `FaceEmbedder` | `face` | `face` | FaceNet identity (InceptionResnetV1, face crops, no text encoder) | 512 |
| `TextE5Embedder` | `e5` | `text` | E5-base-v2 (intfloat/e5-base-v2) | 768 |
| `TextBGEEmbedder` | `bge` | `text` | BGE-base-en-v1.5 (BAAI/bge-base-en-v1.5) | 768 |
| `VideoXClipEmbedder` | `xclip` | `video` | X-CLIP (microsoft/xclip-base-patch32) | 768 |

The image embedders come in **single/patch pairs**: `_single` slugs expose only the CLS-pooled vector (same shape and cost as SigLIP); `_patch` slugs additionally populate `media["patch_grid"]` (raw `H × W × D` fp16) so the region-similarity, region-aware detector scoring, and region-voting code paths can opt in. Both variants of a backbone share weights via an underscore-prefixed `_<backbone>_shared.py` module that the auto-discovery scan skips.

### Embedder capability flags

`MediaEmbedder` carries four capability flags consumed by the routes layer and the frontend. They are **read-only properties on the base class, not plain class attributes** — override them with `@property` on your subclass rather than assigning `supports_text = False` in the class body:

```python
@property
def supports_text(self) -> bool:
    return False
```

| Flag | Default | When to override |
|---|---|---|
| `supports_text: bool` | **True** | Return `False` for vision-only or patch-based encoders with no text tower (DINOv2/v3, EUPE, SIFT/VLAD, FaceNet, BEATs, AST, Whisper-encoder). The default is `True` because most shipped embedders are cross-modal (CLIP, SigLIP, CLAP, E5, BGE); leaving it unset on a vision-only encoder wrongly advertises `POST /api/sort` text queries. |
| `supports_patch_regions: bool` | False | Return `True` for image embedders that implement `_patch_forward_impl(media) -> PatchEmbedOutput` (the hook behind the public `patch_forward`, which takes the `_embed_lock` for you). Loaders that see this flag run the patch pass after the standard CLS pass and store a hierarchical region set plus the raw patch grid. |
| `supports_geometric_verification: bool` | False | Return `True` for structural embedders that produce local features for instance matching (SIFT/VLAD). The loader then asks for a `StructuralFeatures` per image and stores it as `media["local_features"]`, enabling the geometric re-rank and match-stat verification paths. Deliberately media-agnostic, so an audio fingerprint backend can reuse it. |
| `license_notice: Optional[str]` | None | Return a short human-readable string when the upstream weights carry a usage restriction (e.g. FAIR Noncommercial). The frontend embedder picker surfaces this as a warning chip; the dataset-create flow surfaces it inline. |

All four are included in `MediaEmbedder.to_dict()` and exposed via `GET /api/embedders`.

---

## Adding a Media Clipper

Media clippers split a single media item into one or more items of the
**same** type. Unlike processors which return metadata about media,
clippers return **new media dicts** that can replace the original.

**Library contract:**
[`vtscore/docs/extending/clippers.md`](../vtscore/docs/extending/clippers.md)
states the same contract from the library side, and is the guide to follow
when shipping a clipper as a separate distribution rather than adding one to
this repo. This section is the in-repo path: the same contract plus the
app-tier wiring.

### Built-in clippers

| Clipper | Name | Media Type | Description |
|---------|------|------------|-------------|
| `SoundDefaultClipper` | `sound_default` | `audio` | Import each audio file as-is, without splitting |
| `SoundTilingClipper` | `sound_tiling` | `audio` | Split each audio file into fixed-length overlapping segments |
| `SoundSilenceClipper` | `sound_silence` | `audio` | Split each audio file into non-silent segments; drops intro/outro silence |
| `SoundSpeechActivityClipper` | `sound_speech_activity` | `audio` | Split each audio file into one clip per speech turn (Silero VAD) |
| `ImageDefaultClipper` | `image_default` | `image` | Import each image as-is, without splitting |
| `ImageTilingClipper` | `image_tiling` | `image` | Tile each image into equidistant square crops along the longer axis |
| `ImageObjectClipper` | `image_object` | `image` | Detect objects with YOLO/RT-DETR and crop one clip per detection |
| `TextDefaultClipper` | `text_default` | `text` | Import each text entry as-is, without splitting |
| `TextParagraphClipper` | `text_paragraph` | `text` | Split each text entry into paragraphs separated by blank lines |
| `TextSentenceClipper` | `text_sentence` | `text` | Split each text entry into individual sentences |
| `VideoDefaultClipper` | `video_default` | `video` | Import each video as-is, without splitting |
| `VideoTilingClipper` | `video_tiling` | `video` | Split each video into fixed-length overlapping segments |
| `VideoAutoClipper` | `video_auto` | `video` | Tile only when the video is longer than a threshold; pass through otherwise |
| `VideoSceneClipper` | `video_scene` | `video` | Automatically split each video at detected scene changes |
| `DocumentDefaultClipper` | `document_default` | `document` | Import each document as-is, without splitting |
| `FaceDefaultClipper` | `face_default` | `face` | Keep each face crop as-is, without splitting |

**Clipper names carry no parameter suffix.** A parameterised clipper
registers under one stable name (`sound_tiling`, not
`sound_tiling_2.0s`); the parameter values live in the `parameters`
descriptors and travel with the clip's origin as a separate `params`
dict. Two `SoundTilingClipper` instances with different durations would
collide in the registry, so a media type ships one instance per clipper
class and the user re-parameterises it through `with_params()`.

### What to implement

Subclass `MediaClipper` from `vtscore.media.base`.

```python
# vtscore/media/audio/clipper.py  (or a new file)

from vtscore.media.clipper import MediaClipper
from typing import Any


class SoundOverlapClipper(MediaClipper):
    """Tile audio with 50% overlap between segments."""

    def __init__(self, duration: float) -> None:
        self._duration = duration

    @property
    def name(self) -> str:
        """Unique identifier for this clipper (no parameter suffix)."""
        return "sound_overlap"

    @property
    def media_type(self) -> str:
        """The type_id this clipper operates on."""
        return "audio"

    @property
    def description(self) -> str:
        """Short tooltip shown on hover in the clipper chooser UI."""
        return "Tile audio with 50% overlap between consecutive segments."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Split media into one or more media dicts of the same type.

        Each returned dict preserves the original structure (id,
        media_type, category, origin, etc.) but with updated content.
        Returns a list with at least one element.
        """
        wav_bytes = media.get("media_bytes")
        if wav_bytes is None:
            return [media]

        # ... implement overlapping tiling logic ...
        # Return list of new media dicts with updated media_bytes, duration, etc.
        return [media]  # placeholder
```

### Shared helpers

Don't hand-roll what `vtscore/media/clipper.py` already exports.
`clip_with_bounds(media, index, start, end)` returns a copy of *media*
stamped with the standard `duration` / `clip_index` / `clip_start` /
`clip_end` fields (add your own on top); `tile_starts(total, duration,
min_overlap)`, `validate_tiling_params(...)` and
`tiling_parameters(..., item_label=...)` are the tiling arithmetic,
constructor validation and parameter descriptors shared by the audio and
video tiling clippers. A **default** (no-op) clipper subclasses the
concrete `DefaultClipper` base instead of `MediaClipper`, passing its
name, media type and description to `super().__init__()` — see
[`vtscore/docs/extending/clippers.md § Shared
helpers`](../vtscore/docs/extending/clippers.md#shared-helpers).

For audio specifically, `_emit_wav_segments` in
`vtscore/media/audio/clipper.py` turns a list of `(start, end)` ranges
into clip dicts with the WAV bytes sliced and `file_size` refreshed, so a
new audio clipper only has to produce the ranges.

### Register the clipper

Add the clipper to the `CLIPPERS` sentinel list in your media type's
`__init__.py`:

```python
# vtscore/media/audio/__init__.py

from vtscore.media.audio.clipper import SoundOverlapClipper
# ...
CLIPPERS = [
    SoundTilingClipper(10.0, 1.0),
    SoundDefaultClipper(),
    SoundSilenceClipper(),
    SoundSpeechActivityClipper(),
    SoundOverlapClipper(2.0),  # new
]
```

Each entry is registered under `clipper.name`, so list **one instance
per clipper class** — a second instance of the same class would
overwrite the first.

### MediaClipper abstract interface reference

**Required abstract properties:**

| Property     | Returns | Description                                    |
|--------------|---------|------------------------------------------------|
| `name`       | `str`   | Unique identifier, with no parameter suffix (e.g. `"sound_tiling"`) |
| `media_type` | `str`   | The `type_id` this clipper operates on          |

**Required abstract methods:**

| Method          | Signature              | Description                        |
|-----------------|------------------------|------------------------------------|
| `clip(media)`   | `(dict) -> list[dict]` | Split one media into one or more   |

**Optional overridable methods/properties:**

| Method/Property      | Signature / Returns      | Description                                                       |
|----------------------|--------------------------|-------------------------------------------------------------------|
| `display_name`       | `str`                    | Human-readable name for UI tabs (default: `name` minus its type prefix, title-cased) |
| `description`        | `str`                    | Short tooltip text shown on hover in the clipper chooser UI       |
| `summary_template`   | `str`                    | One-line preview with `{key}` placeholders for parameter values (defaults to `description`) |
| `to_dict()`          | `() -> dict`             | JSON-serialisable metadata (default: name + display_name + media_type) |
| `parameters`         | `list[dict[str, Any]]`   | Configurable parameters (key, label, type, default, description)  |
| `creation_questions` | `list[dict[str, Any]]`   | Questions shown at creation time (defaults to `parameters`)       |
| `with_params(p)`     | `(dict) -> MediaClipper` | Return a **new** clipper with overridden parameters; never mutate `self` |
| `resolve_for_durations(d)` | `(list[float]) -> MediaClipper` | **Reserved - never called.** An override here is inert; use `resolve_for_media` |
| `resolve_for_media(m)` | `(dict) -> MediaClipper` | Per-media hook; used by auto-selecting clippers (e.g. `video_auto`) |

Parameter dicts support an optional `description` key alongside `label`
(shown as a tooltip when the user hovers over the setting in
the clipper chooser dialog).

### Clip method contract

Each dict in the returned list must:
- Preserve the structure of the original (`id`, `media_type`,
  `category`, `origin`, `origin_name`, etc.)
- Contain the clipped content (updated `media_bytes`/`media_string`,
  `duration`, and any type-specific fields)
- Default clippers return `[media]` unchanged

---

## Adding a Media Cleaner

Media cleaners remove **content-free regions** from an item so the embedder
spends its representational capacity on signal instead of letterbox bars,
leading silence, or PDF-extraction junk. Like a clipper a cleaner maps type X
to type X; it differs in **cardinality** and **use**:

|             | Clipper                    | Converter     | Cleaner                                       |
|-------------|----------------------------|---------------|-----------------------------------------------|
| Type        | X → X                      | X → Y         | X → X                                         |
| Cardinality | 1 → N                      | 1 → N         | **1 → 1**                                     |
| UI          | pick **one** per import    | routing step  | **all optional gates**, independently toggled |

A clipper breaks large media into manageable sub-items; a cleaner tightens each
item in place. Cleaners therefore run **after the final clipper/converter
step**, on the units that will actually be embedded, and only the cleaners
matching the chain's *final* media type apply (a document→text chain gets text
cleaners, not document cleaners). They run in registration order, with no user
reordering, so every shipped cleaner should be order-insensitive in practice.

`MediaCleaner` subclasses `MediaClipper`, so the whole descriptor stack
(`name` / `media_type` / `display_name` / `description` / `parameters` /
`creation_questions` / `with_params` / `to_dict`) is inherited unchanged, and a
cleaner rides the existing clipper chain as an `n_out == 1` step. Cleaners live
in their **own registry**, so they never appear in a clipper chooser:
`GET /api/cleaners` lists them and the import form renders one checkbox per
entry.

That checkbox list lives **strictly behind the Add Dataset modal's "Advanced ▾"
toggle** and never escapes it. This is deliberate and differs from the embedder
and clipper pickers, which stay on screen with Advanced collapsed once the user
overrides them: cleanup is the most technical knob in the modal, so a
non-default selection does *not* pull the block back into view — it is surfaced
only in the Advanced toggle's tooltip (`Cleanup: <enabled gates>`). Because the
block cannot be reached any other way, registering a cleaner for a media type
also forces the Advanced toggle itself to render, even in flows that would
otherwise hide it. Do not add a cleaner affordance outside that block.

### Built-in cleaners

| Cleaner | Name | Media Type | Default | Description |
|---------|------|------------|---------|-------------|
| `ImageExifOrientCleaner` | `image_exif_orient` | `image` | off | Bake a photo's EXIF display orientation into its stored bytes (VTSearch already *reads* every image upright; this is for tools that ignore EXIF) |
| `ImageEdgeTrimCleaner` | `image_edge_trim` | `image` | off | Crop near-solid white/black margins (letterbox, pillarbox, whitespace around a logo) |
| `AudioSilenceTrimCleaner` | `audio_silence_trim` | `audio` | off | Drop the silence at the head and tail of a clip, keeping internal pauses |
| `TextMarkupStripCleaner` | `text_markup_strip` | `text` | off | Remove HTML tags and Markdown syntax, keeping the text inside |
| `TextWhitespaceCleaner` | `text_whitespace` | `text` | off | Collapse whitespace runs, drop control characters, rejoin hyphen-broken words |
| `VideoLetterboxCropCleaner` | `video_letterbox_crop` | `video` | off | Record the letterbox / pillarbox crop every sampled frame agrees on as the unit's `clip_box` |
| `VideoBlankTrimCleaner` | `video_blank_trim` | `video` | off | Narrow the unit's `clip_start` / `clip_end` past its blank head and tail (black leader, fade-ins, empty tail cards) |

Three of these share their detector with another caller rather than owning a
second copy of the heuristic: `image_edge_trim`, `video_letterbox_crop`, and the
grid thumbnail all call `vtscore/media/image/edge_trim.py`, and
`audio_silence_trim` and `SoundSilenceClipper` both call
`vtscore/media/audio/silence.py`. When a cleaner answers a question something
else in the codebase already answers, extract the detector; a cleaner that
disagrees with the thumbnail about where the content starts is worse than no
cleaner.

### What to implement

Subclass `MediaCleaner` from `vtscore.media.cleaner` and implement `clean()`;
the inherited `clip()` wraps it as a single-output chain step.

```python
# vtscore/media/text/cleaner.py

from typing import Any

from vtscore.media.cleaner import MediaCleaner


class TextWhitespaceCleaner(MediaCleaner):
    """Collapse whitespace runs and strip control characters."""

    @property
    def name(self) -> str:
        return "text_whitespace"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def description(self) -> str:
        """Shown on hover next to the cleanup checkbox."""
        return "Collapse whitespace runs and strip control characters."

    @property
    def default_enabled(self) -> bool:
        """Whether the import form pre-checks this cleaner's box.

        ``False`` (the default) for anything that makes a judgment call about
        what counts as wasted content. Return ``True`` only when leaving the
        gate off means shipping known-wrong vectors.
        """
        return False

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        text = media.get("media_string")
        if not isinstance(text, str):
            return media
        collapsed = " ".join(text.split())
        if collapsed == text:
            return media          # nothing to clean: return *media* itself
        cleaned = dict(media)     # copy, never mutate in place
        cleaned["media_string"] = collapsed
        cleaned["character_count"] = len(collapsed)
        return cleaned
```

### Register the cleaner

Add it to the `CLEANERS` sentinel list in your media type's `__init__.py`:

```python
# vtscore/media/text/__init__.py

from vtscore.media.text.cleaner import TextWhitespaceCleaner
# ...
CLEANERS = [TextWhitespaceCleaner()]
```

### Clean method contract

- **Return the media unchanged** (the *same* dict, or an equal copy) when there
  is nothing to clean or the payload can't be decoded. Like a clipper, a cleaner
  never aborts a load; a degenerate input is a no-op, not an error. Be
  conservative by construction — cap how much you are willing to remove.
- **Never mutate the input in place.** Build the output with `dict(media)` and
  overwrite the payload keys. The runner detects "nothing changed" by comparing
  the payload (and the window / box keys, see below) before and after, and an
  in-place mutation leaves it no pre-clean version to keep.
- **Update the metadata you invalidated** — `file_size`, `width` / `height`,
  `duration`, `character_count`. MD5, embeddings, and thumbnails are redone for
  you (see below).

### The dual payload: Clean vs Original

The cleaned payload becomes the **canonical** content: `media_bytes` /
`media_string`, `duration`, `file_size`, MD5, thumbnail, and the embedding all
derive from it, so every existing consumer works unchanged.

The chain **runner** — not each cleaner — additionally snapshots the pre-clean
payload the first time a cleaner actually changes an item, under
`original_media_bytes` / `original_media_string` / `original_duration`:

- Only a real change stores anything. Most cleaners no-op on most items, which
  bounds the storage cost well below a blanket 2×.
- With several cleaners in sequence the snapshot is taken **once**, before the
  first mutating gate, so "Original" always means the pre-*any*-clean payload.
- Dataset pickles persist it: it is the only copy of what the user imported, so
  it is dataset *content*, not a cache (the standing pickle exception to the
  no-persisted-artifacts rule). Embeddings are still derived only from the
  canonical payload.
- The media payload gains a `has_original` flag, the byte / text routes accept
  `?variant=original`, and the detail viewer shows a Clean/Original toggle when
  the flag is set.
- Because a cleaner rewrites the payload rather than slicing it, a cleaned item
  from a *reference* (thin) import keeps its bytes materialized — `lazy_clip`
  has no recipe that reproduces a cleaner's output from the source file.

A cleaned item is flagged for MD5 + embedding + thumbnail recomputation, and its
trail entry records `changed`, so provenance's "Derived Via" line lists only the
gates that actually did something to that item.

### Cleaning by metadata instead of payload

A **video** unit is a `(parent bytes, time window)` pair: every clip of a tiled
video shares the parent's payload and says which slice of it it is via
`clip_start` / `clip_end`. A video cleaner therefore cleans by *narrowing
metadata* rather than by rewriting bytes — re-encoding a cleaned copy per unit
would duplicate the payload once per tile and desync the window, which still
indexes the parent's timeline.

The runner treats a change to any of `clip_start`, `clip_end`, `clip_box`
(`CLEANED_METADATA_KEYS` in `vtscore/datasets/clipper_chain.py`) as a real
change: the item is flagged for MD5 + embedding + thumbnail recomputation, its
trail entry records `changed` plus the new window / box, and its stale
thumbnail is dropped. What it does **not** do is snapshot an `original_*`
payload — there is no second payload, the served file is untouched, and the
player already loops within `[clip_start, clip_end]`. Two consequences, both
good: a metadata-cleaned unit costs no extra storage, and a reference (thin)
import keeps its byte savings on it.

The spatial half of that contract, `clip_box`, is *honoured* rather than baked
in, by three readers that have to agree (the parsing and the crop live once in
`vtscore/media/video/crop.py`):

| Reader | Why it needs the box |
|--------|---------------------|
| `sample_video_frames` (`vtscore/media/video/_frame_sampling.py`) | what the embedder actually sees |
| the video thumbnailers (`vtscore/media/video/media_type.py`) | so the grid preview frames the same picture |
| `_fixup_clip_md5_and_embeddings` (`vtscore/datasets/stages/clipper.py`) | folds the box into the boundary-tag MD5 so two crops of one parent don't collide |

If you add a media type whose units are likewise metadata-only, follow the same
shape: narrow the keys, let the readers honour them, and snapshot nothing.

---

## Adding a Media Converter

Media converters transform content from one media type to another (e.g.
document pages to images, video to audio). Converters are
**auto-discovered** via `PluginRegistry` with the `CONVERTER` sentinel,
just like other plugin families.

**Library contract:**
[`vtscore/docs/extending/converters.md`](../vtscore/docs/extending/converters.md)
states the same contract from the library side, and is the guide to follow
when shipping a converter as a separate distribution rather than adding one
to this repo. This section is the in-repo path: the same contract plus the
app-tier wiring.

### Built-in converters

<!-- BEGIN GENERATED: plugins:converters -->
<!-- Generated by scripts/gen-docs-inventories.py; do not edit by hand. Refresh with: python scripts/gen-docs-inventories.py -->

| Converter | Conversion | Display name | Description |
|---|---|---|---|
| `audio2image` | `audio` → `image` | Audio → Image (spectrogram) | Render audio as a mel-spectrogram or CQT image |
| `audio2text` | `audio` → `text` | Audio → Text (Whisper ASR) | Transcribe speech in audio to text via Whisper |
| `document2image` | `document` → `image` | Document → Images | Render document pages as images |
| `document2text` | `document` → `text` | Document → Text | Extract embedded text from documents |
| `image2face` | `image` → `face` | Images → Faces | Detect faces in images and crop one face per detection |
| `image2text` | `image` → `text` | Image → Text (OCR) | Extract text from images via OCR |
| `video2audio` | `video` → `audio` | Video → Audio | Extract audio tracks from video files |
| `video2image` | `video` → `image` | Video → Images | Extract frames from video files |

<!-- END GENERATED: plugins:converters -->

### File structure

```
vtscore/converters/<source>2<target>.py   # Your converter class
```

### What to implement

Subclass `MediaConverter` from `vtscore.converters.base`.

```python
# vtscore/converters/audio2text.py

from vtscore.converters.base import MediaConverter
from vtscore.plugins import PluginField
from typing import Any


class Audio2TextMediaConverter(MediaConverter):

    display_name = "Audio → Text"
    description = "Transcribe audio to text using a speech model."
    fields = [
        PluginField(
            key="language",
            label="Language",
            field_type="text",
            default="en",
            description="ISO language code passed to the speech model.",
        ),
    ]

    @property
    def source_type(self) -> str:
        """The type_id of the input media type."""
        return "audio"

    @property
    def target_type(self) -> str:
        """The type_id of the output media type."""
        return "text"

    def convert(
        self,
        media: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert one media dict into one or more target-type media dicts.

        Each returned dict must include:
        - "filename": a descriptive filename
        - Data fields expected by the target type (e.g. "media_string"
          for text, "media_bytes" for images)

        Returns empty list if conversion fails.
        Does NOT include "id", "embedding", or "md5"; the caller handles those.
        """
        language = params["language"]
        # ... transcription logic ...
        return [{"filename": "transcript.txt", "media_string": transcript}]
```

**The `params` parameter is not optional in practice.** The abstract
method is `convert(self, media, params=None)`, and every framework call
site reaches it through `convert_normalized()`, which validates `params`
against your `fields` schema and fills every declared key with its
`default` before dispatching. So inside `convert()` you can index
`params[key]` directly; the `None` default exists only for third-party
callers who invoke `convert()` by hand (they should call
`convert_normalized()` instead, or read through the
`get_param(params, key)` shim). A subclass that declares
`convert(self, media)` — without the second parameter — raises
`TypeError` on the first conversion.

Use `resolve_media_bytes(media)` (also in `vtscore.converters.base`)
rather than reading `media["media_bytes"]` directly: reference (*thin*)
imports hand the converter only `{filename, media_path}`, so a converter
that reads `media_bytes` alone silently produces nothing for every thin
import.

### Register the converter

Expose a `CONVERTER` sentinel at module level in your converter file:

```python
# At the bottom of vtscore/converters/audio2text.py

CONVERTER = Audio2TextMediaConverter()
```

The `PluginRegistry` auto-discovers `.py` files in `vtscore/converters/`
that expose a `CONVERTER` attribute. No manual registration in
`__init__.py` is needed.

<!--
   Old explicit registration (no longer needed):
   ```python
   __all__ = [
       # ... existing entries ...
       "Audio2TextMediaConverter",
   ]
   ```
-->

### MediaConverter abstract interface reference

**Required abstract properties:**

| Property      | Returns | Description                                |
|---------------|---------|--------------------------------------------|
| `source_type` | `str`   | The `type_id` of the input media type      |
| `target_type` | `str`   | The `type_id` of the output media type     |

**Required abstract methods:**

| Method                        | Signature                             | Description                              |
|-------------------------------|---------------------------------------|------------------------------------------|
| `convert(media, params=None)` | `(dict, dict \| None) -> list[dict]`  | Convert one media into target-type dicts |

**Optional class attributes:**

| Attribute               | Type                | Default | Description                              |
|-------------------------|---------------------|---------|------------------------------------------|
| `display_name`          | `str`               | `""`    | Human-readable label (auto-derived if empty) |
| `description`           | `str`               | `""`    | Short description of the conversion      |
| `summary_template`      | `str`               | `""`    | One-line preview with `{key}` placeholders for parameter values; shown on the import row (falls back to `description`) |
| `fields`                | `list[PluginField]` | `[]`    | User-configurable parameters, delivered to `convert()` as `params` |

**Framework-owned methods (call, don't override):**

| Method                                   | Description |
|------------------------------------------|-------------|
| `convert_normalized(media, params=None)` | The framework entry point: validates `params` against `fields`, default-fills every declared key, then calls `convert()`. Raises `ValueError` on a bad value. Third-party call sites should use this rather than `convert()`. |
| `normalize_params(params)`               | Validate + default-fill only; `convert_normalized()` wraps it. |
| `get_param(params, key)`                 | Read one param with a `default` fallback. A shim for converters whose callers bypass `convert_normalized()`; framework-routed calls can just index `params[key]`. |

**Derived property (not overridable):**

| Property | Returns | Description |
|----------|---------|-------------|
| `name`   | `str`   | Auto-generated as `"{source_type}2{target_type}"` |

---

## Adding a Media Source

Media sources provide low-level access to media files at a location
(local folder, HTTP archive, S3 bucket, etc.). They sit *below* dataset
importers. Importers that access file-like storage compose a
`MediaSource` for single-file resolution and cross-dataset label
re-ingestion.

Sources are **stateful** (e.g. an archive source may download and extract
on first access), so each call to `get_source_for_origin()` returns a
fresh instance. Callers should call `cleanup()` when done.

### File structure

```
vtscore/datasets/sources/<your_source>.py     # Source factory + SOURCE instance (required)
```

Media sources are **flat `.py` modules**, not sub-packages — the discovery scan
(`discover_modules=True` in `vtscore/datasets/sources/__init__.py`) walks module
files and picks up their `SOURCE` sentinel. A source built inside a subdirectory
`__init__.py` is never seen. Every built-in source (`local_folder.py`,
`http_archive.py`, `server_files.py`, …) follows this shape.

### What to implement

Unlike other plugin families, media sources use a **factory pattern**.
The `SOURCE` sentinel is a factory object with a `create_from_origin()`
method that returns a `MediaSource` instance.

All fetch and resolve methods return `FetchedItem` instead of a bare
`Path | None`. `FetchedItem` carries the local path alongside optional
pre-computed data that the source's API may have returned alongside the
file — embeddings, embedder name, file size, duration, and so on. The
ingest path uses whatever is in `FetchedItem` to avoid redundant local
work (re-embedding from disk, re-reading for file size, etc.).

```python
# vtscore/datasets/sources/s3.py

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource


class S3MediaSource(MediaSource):
    """Access media files in an S3 bucket."""

    name = "s3"

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self._bucket = bucket
        self._prefix = prefix

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        """Yield all media items in the bucket (optionally filtered by extension)."""
        import boto3
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.rsplit("/", 1)[-1]
                if extensions and not any(filename.lower().endswith(e) for e in extensions):
                    continue
                yield MediaItem(key=key, filename=filename, source_name=self.name)

    def fetch_item(self, key: str) -> FetchedItem:
        """Download an item to a temp directory and return a FetchedItem."""
        import boto3
        local = Path(tempfile.gettempdir()) / "vtsearch_s3" / key
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            boto3.client("s3").download_file(self._bucket, key, str(local))
        return FetchedItem(path=local)

    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
        """Resolve a media file by origin_name or filename."""
        for candidate in (origin_name, filename):
            if candidate:
                key = f"{self._prefix}{candidate}" if self._prefix else candidate
                item = self.fetch_item(key)
                if item.path and item.path.exists():
                    return item
        return FetchedItem(path=None)

    def fetch_items(self, keys: list[str]) -> dict[str, FetchedItem]:
        """Download multiple items concurrently instead of one at a time.

        Override the default loop to use a thread pool. If your backing
        service returns embeddings or metadata alongside the file bytes,
        populate ``FetchedItem.embedding`` / ``FetchedItem.extra`` here
        so the ingest path can skip local re-embedding and stat calls.
        """
        def _fetch(key: str) -> tuple[str, FetchedItem]:
            return key, self.fetch_item(key)

        with ThreadPoolExecutor(max_workers=8) as pool:
            return dict(pool.map(_fetch, keys))

    def resolve_paths(
        self, entries: list[tuple[str, str]]
    ) -> list[FetchedItem]:
        """Resolve multiple entries concurrently.

        The ingest fast-path calls ``resolve_paths`` (not ``fetch_items``),
        so override this to parallelise the path used by re-ingestion.
        """
        def _resolve(pair: tuple[str, str]) -> FetchedItem:
            return self.resolve_path(*pair)

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(_resolve, entries))


class _S3SourceFactory:
    """Factory that creates S3MediaSource instances from origin dicts."""

    name = "s3"

    def create_from_origin(self, origin: dict[str, Any]) -> S3MediaSource | None:
        params = origin.get("params", {})
        bucket = params.get("bucket", "")
        if not bucket:
            return None
        return S3MediaSource(bucket, params.get("prefix", ""))


SOURCE = _S3SourceFactory()
```

### FetchedItem fields

Every fetch and resolve method returns `FetchedItem`. The `path` field is
always required; the rest are optional bonuses that let sources surface
data they already have without forcing VTSearch to re-derive it.

| Field | Type | When to set |
|-------|------|-------------|
| `path` | `Path \| None` | Always — local file path, or `None` if not found/downloadable |
| `embedding` | `np.ndarray \| None` | When your API returns a pre-computed vector. The ingest path skips re-embedding when this is present (provided the origin has no clip params) |
| `embedder_name` | `str` | Required alongside `embedding` — names the embedding space so vectors can be matched against the dataset |
| `extra` | `dict[str, Any]` | Source-authoritative field overrides written into the media record (e.g. `{"file_size": 12345, "duration": 3.5, "created_at": "2024-01-01T00:00:00Z"}`) |

For a simple source that just downloads files, returning
`FetchedItem(path=local_path)` is sufficient. The other fields are only
worth populating when your backing API already provides them in the same
round-trip.

### MediaSource abstract interface reference

**Required abstract methods** — must return `FetchedItem`, not `Path | None`:

| Method | Signature | Description |
|--------|-----------|-------------|
| `list_items()` | `(extensions: list[str] \| None) -> Iterator[MediaItem]` | Yield all media items, optionally filtered |
| `fetch_item()` | `(key: str) -> FetchedItem` | Fetch by key; `item.path` is `None` if not found |
| `resolve_path()` | `(origin_name: str, filename: str) -> FetchedItem` | Find by origin_name or filename; `item.path` is `None` if not found |

**Optional methods** — default implementations loop over the single-item
counterparts; override for parallelism or to surface pre-computed data:

| Method | Signature | Description |
|--------|-----------|-------------|
| `fetch_items()` | `(keys: list[str]) -> dict[str, FetchedItem]` | Bulk form of `fetch_item`. Override to parallelise downloads or return pre-computed embeddings/metadata |
| `resolve_paths()` | `(entries: list[tuple[str, str]]) -> list[FetchedItem]` | Bulk form of `resolve_path`, result aligned with input. The ingest fast-path calls this; override to parallelise re-ingestion |
| `cleanup()` | `() -> None` | Release temporary resources (default: no-op) |

**Data types:**

| Type | Fields | Description |
|------|--------|-------------|
| `MediaItem` | `key`, `filename`, `source_name` | A discoverable file within a source (yielded by `list_items`) |
| `FetchedItem` | `path`, `embedding`, `embedder_name`, `extra` | Result of any fetch or resolve call; `path` may be `None` |

### How it gets invoked

`get_source_for_origin(origin_dict)` looks up the factory by matching
`origin["importer"]` to the factory's `name`, then calls
`factory.create_from_origin(origin)`.

The ingest fast-path (`vtscore.datasets.ingest._ingest_via_source`) calls
`resolve_paths()` once for all missing entries before the sequential
embed loop, so any parallelism in your `resolve_paths()` override
completes before per-item work begins. If your source's `resolve_path()`
populates `FetchedItem.embedding`, the embed step is skipped automatically.

---
