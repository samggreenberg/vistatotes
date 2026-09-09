# Writing a `MediaEmbedder`

A media embedder turns a media file (or a text query) into a fixed-size
vector in some embedding space. Each embedder belongs to exactly one
media type, but a type can have multiple embedders - see the generated
roster in [`docs/ML.md` § Embedding
Models](../../../docs/ML.md#embedding-models). Embedders
are auto-discovered: any `embedder_<name>.py` file (or `embedder_<name>/`
sub-package) inside a media-type sub-package gets imported, and its
module-level `EMBEDDER` sentinel is registered. Subclass
[`MediaEmbedder`](../../media/embedder.py)
([`vtscore/media/embedder.py`](../../media/embedder.py)), implement
two abstract methods, and expose the sentinel.

**App-side counterpart:** [`docs/EXTENDING-media.md § Adding a Media
Embedder`](../../../docs/EXTENDING-media.md#adding-a-media-embedder).
This guide focuses on the library API, the lazy-load pattern, and the
batch / bulk hooks.

## Contents

- [The contract](#the-contract)
- [Where the file goes](#where-the-file-goes)
- [Lazy model loading](#lazy-model-loading)
- [Embedding hooks: single, bulk, patch](#embedding-hooks-single-bulk-patch)
- [Capability flags](#capability-flags)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## The contract

`MediaEmbedder` is an ABC. Subclasses must implement:

| Member | Type | Purpose |
|--------|------|---------|
| `name` (property) | `str` | Unique registry key - `"clap"`, `"siglip"`, `"my_embedder"` |
| `media_type_id` (property) | `str` | The `MediaType.type_id` this embedder works on |
| `_load_models_impl()` | `() -> None` | Idempotent model load - override this, NOT `load_models()` |
| `_embed_media_impl(media)` | `(dict) -> np.ndarray \| None` | Embed one media dict - override this, NOT `embed_media()` |

Optional but commonly overridden:

| Member | Default | Purpose |
|--------|---------|---------|
| `display_name` | `name` | Friendlier label for the picker UI |
| `is_default` | `False` | Exactly one embedder per media type should return `True` - that one is what `embedders_for_type(t)[0]` returns |
| `_embed_text_impl(text)` | `None` | Text-query hook - override this, NOT `embed_text`, which L2-normalizes the result for you. Leave the default to disable text-query sort; otherwise return a vector in the same space as `_embed_media_impl` |
| `description_wrappers` | `[]` | Templates with `{text}` for enriched text embedding (e.g. `["the sound of {text}"]`). Keep the default unless you have measured that the ensemble beats the typed query on your checkpoint - it is a loss on most (#3127/#3341) |
| `_embed_media_bulk_impl(medias)` | per-item loop | Native bulk hook for service embedders or batched GPU forward passes |
| `_patch_forward_impl(media)` | `None` | For patch-capable image encoders; required when `supports_patch_regions = True` |
| `models_loaded()` | `self._model is not None` | Whether the weights are already resident. Callers that must *plan around* the load read this - the text-sort progress bar drops its model-load step entirely when it is `True`. Override it only if you hold the backbone somewhere other than `self._model` (override `loaded_backbone()` in the same breath); the default's permanent `False` is safe but wrong |

Don't override the public methods `embed_media`, `embed_text`,
`embed_media_bulk`, `load_models`, or `patch_forward` - they wrap the
`_impl` hooks with shared locking, L2-normalization, and progress
dispatch.

## Where the file goes

In-tree:

```
vtscore/media/<media_type>/embedder_<name>.py
```

The discovery scan ([`vtscore/media/__init__.py`](../../media/__init__.py))
looks for any file matching `embedder*.py` (or any directory matching
`embedder*/` with an `__init__.py`) inside a media-type sub-package
and registers its `EMBEDDER` sentinel.

Out-of-tree, the simplest path is to symlink your `embedder_<name>.py`
into the right directory - the scan uses
`importlib.util.spec_from_file_location` so symlinked files load
cleanly. There is no `vtscore.embedders` entry-point group; embedders
are tied to a specific media-type package by directory placement.

## Lazy model loading

Embedders are instantiated at import time but should not load model
weights until they're first needed. The pattern:

```python
class MyEmbedder(MediaEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self._model = None  # populated on first call

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return  # idempotent - concurrent callers all wait on the per-class lock
        from somewhere import HeavyModel  # imports inside method = lazy
        cache_dir = embedder_load_setup(self._on_progress, "Loading MyModel weights…")
        self._model = HeavyModel.from_pretrained("…", cache_dir=cache_dir)
```

Three helpers in [`vtscore/media/load_progress.py`](../../media/load_progress.py),
re-exported from [`vtscore.media.embedder`](../../media/embedder.py) along with
everything else in that module, keep heavy loads non-blocking from the UI's
point of view:

- `embedder_load_setup(on_progress, message)` - calls
  `ensure_torch_configured()`, emits an initial progress event, and
  returns the model cache directory as a string. Use it at the top of
  every `_load_models_impl()`.
- `timed_progress(on_progress, status, message, ...)` - context
  manager that appends an elapsed-time suffix to a long-running
  import or load step every second, so the UI doesn't look frozen.
  Pass `est_modules=IMPORT_MODULE_ESTIMATES["<lib>"]` to *drive* the
  bar from the live `sys.modules` delta: it starts at 0 %, climbs as
  submodules load, and is clamped below 100 % until the import returns
  (so a still-importing step never reads as done). Without it the
  passed `current`/`total` are forwarded unchanged.
- `load_pretrained_local_first(load_fn, *args, **kw)` - tries
  `local_files_only=True` first so a slow Hub round-trip doesn't
  stall a cached model load, then falls back to a network load with
  retry / backoff for transient 5xx errors.

The framework writes downloaded weights to
`vtscore.config.MODELS_CACHE_DIR` (which honours `VTSEARCH_MODELS_DIR`,
fallback `data/models`). Library callers running outside an app shim
should reference this constant directly:

```python
from vtscore.config import MODELS_CACHE_DIR
cache_dir = str(MODELS_CACHE_DIR)
```

Do **not** add a hardcoded `data/models/` path or invent your own
cache directory.

## Embedding hooks: single, bulk, patch

`embed_media(media)` acquires the global `_embed_lock` and dispatches
to `_embed_media_impl(media)`. The lock serialises every forward pass
across every embedder type, which prevents two CLIP models from
fighting over GPU memory.

`embed_media_bulk(medias)` is the bulk entry point. The default
`_embed_media_bulk_impl` loops over `embed_media` and emits per-item
progress through `self._on_progress("embedding", ..., i, total)` - fine
for embedders that don't batch internally. Override
`_embed_media_bulk_impl` when:

- the underlying model accepts a batch tensor and you want to fuse N
  forward passes into one;
- the source is a remote service that accepts many items per request;
- you want to issue concurrent fetches via a thread pool.

Overrides that batch internally are responsible for emitting their own
progress updates. `embed_medias(dict[int, dict])` is a sugar wrapper
that delegates to `embed_media_bulk` and returns id-keyed vectors -
importers usually call this.

`patch_forward(media)` is for patch-based image encoders (DINOv2,
DINOv3, EUPE). Set `supports_patch_regions = True` and implement
`_patch_forward_impl(media)` returning a `PatchEmbedOutput`
([`vtscore/media/patch_embed.py`](../../media/patch_embed.py)). The
dataset loader gates on the flag - single-vector embedders leave the
default in place and the patch pipeline skips their datasets entirely.

## Capability flags

Three flags on the subclass change framework behaviour:

| Flag | Default | When to set |
|------|---------|-------------|
| `supports_text` | `True` | Set `False` for vision-only or patch-only encoders that don't have a text tower. The frontend hides text-search affordances for datasets using them. |
| `supports_patch_regions` | `False` | Set `True` for patch-based image encoders that implement `_patch_forward_impl`. The dataset loader runs the patch pass after the standard CLS pass. |
| `license_notice` | `None` | Non-`None` when the upstream weights carry a usage restriction (e.g. FAIR Noncommercial). The frontend surfaces this as a warning chip in the picker. |

All three appear in `MediaEmbedder.to_dict()` and the
`GET /api/embedders` payload.

## Worked example

A minimal text embedder wrapping a `sentence-transformers` model. The
embedder lives alongside the existing text type (`vtscore/media/text/`)
and ships its own embedder package or symlinks in.

```python
# vtscore/media/text/embedder_minilm.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from vtscore.media.embedder import (
    IMPORT_MODULE_ESTIMATES,
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    load_pretrained_local_first,
    timed_progress,
)


class TextMiniLMEmbedder(MediaEmbedder):
    """all-MiniLM-L6-v2 (sentence-transformers) - 384-dim text encoder."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    # -- Identity ------------------------------------------------------

    @property
    def name(self) -> str:
        return "minilm"

    @property
    def display_name(self) -> str:
        return "MiniLM (general text)"

    @property
    def media_type_id(self) -> str:
        return "text"

    @property
    def is_default(self) -> bool:
        return False  # E5 stays default; this is an alternative

    # -- Model lifecycle ----------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return
        with timed_progress(
            self._on_progress,
            "loading",
            "Importing sentence-transformers…",
            est_modules=IMPORT_MODULE_ESTIMATES["sentence_transformers"],
        ):
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        cache_dir = embedder_load_setup(self._on_progress, "Loading MiniLM weights…")
        with intercept_tqdm_progress(self._on_progress):
            self._model = load_pretrained_local_first(
                SentenceTransformer,
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=cache_dir,
            )

    # -- Embedding -----------------------------------------------------

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        text = media.get("media_string")
        if text is None and media.get("media_path"):
            text = Path(media["media_path"]).read_text(errors="replace")[:8000]
        if not text:
            return None
        return self._model.encode(text, normalize_embeddings=True)

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        return self._model.encode(text, normalize_embeddings=True)


EMBEDDER = TextMiniLMEmbedder()
```

That's the full library-side contract. Drop the file into
`vtscore/media/text/embedder_minilm.py` and the next import of
`vtscore.media` auto-discovers it. No `__init__.py` edits needed -
the per-file embedder scan handles registration.

## Testing pattern

Library tests for embedders live in `tests_lib/detectors/`. The
autouse `_stub_embedding_models` fixture in
[`tests_lib/conftest.py`](../../../tests_lib/conftest.py) stubs every
built-in embedder so test runs don't download model weights;
**your test must construct your real embedder explicitly** if you
want to exercise its load path.

```python
# tests_lib/detectors/test_minilm_embedder.py
import numpy as np
import pytest

from vtscore.media import embedders_for_type, get_embedder


class TestMiniLMRegistration:
    def test_is_registered(self):
        names = [e.name for e in embedders_for_type("text")]
        assert "minilm" in names

    def test_is_not_default(self):
        emb = get_embedder("minilm")
        assert emb.is_default is False


class TestMiniLMEmbed:
    @pytest.mark.skipif(
        not _has_module("sentence_transformers"),
        reason="sentence-transformers not installed",
    )
    def test_embed_text_returns_vector(self):
        from my_pkg.embedder_minilm import TextMiniLMEmbedder

        emb = TextMiniLMEmbedder()
        vec = emb.embed_text("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        # Normalized
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None
```

See [`tests_lib/detectors/test_new_embedders.py`](../../../tests_lib/detectors/test_new_embedders.py)
for a longer reference covering registration, capability flags, and
basic embed behaviour. Network-dependent or GPU-only paths belong
under `tests_lib/gpu/` (excluded by default; run with `-m gpu`).
