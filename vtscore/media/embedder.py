"""Media embedder ABC.

The :class:`MediaEmbedder` contract every embedder subclasses, plus the
thread-local progress plumbing behind :attr:`MediaEmbedder._on_progress`.

The two subsystems an embedder *uses* while loading a model live beside this
module and are re-exported below, so a third-party embedder can keep importing
every name it always did from ``vtscore.media.embedder``:

* :mod:`vtscore.media.load_progress` — progress interception during a load
  (``timed_progress``, ``intercept_tqdm_progress``,
  ``intercept_weight_loading_progress``, ``embedder_load_setup``,
  ``load_pretrained_local_first``, ``hf_token``, ``IMPORT_MODULE_ESTIMATES``).
* :mod:`vtscore.media.torch_ops` — tensor/device adapters
  (``extract_tensor``, ``to_compute_device``, ``embed_autocast``,
  ``to_model_inputs``, ``to_float32``).
"""

from __future__ import annotations

import contextlib
import os
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtscore.media.base import ProgressCallback, _noop_progress

# Re-exported for third-party embedders, which the extension guide documents as
# importing these from this module (`vtscore/docs/extending/embedders.md`).
from vtscore.media.load_progress import (
    IMPORT_MODULE_ESTIMATES,
    embedder_load_setup,
    hf_token,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)
from vtscore.media.torch_ops import (
    embed_autocast,
    extract_tensor,
    to_compute_device,
    to_float32,
    to_model_inputs,
)

if TYPE_CHECKING:
    from vtscore.media.patch_embed import PatchEmbedOutput
    from vtscore.media.structural import StructuralFeatures

__all__ = [
    "DEFAULT_EMBED_BATCH_SIZE",
    "IMPORT_MODULE_ESTIMATES",
    "MediaEmbedder",
    "embed_autocast",
    "embedder_load_setup",
    "extract_tensor",
    "hf_token",
    "intercept_tqdm_progress",
    "intercept_weight_loading_progress",
    "load_pretrained_local_first",
    "media_from_path",
    "resolve_embed_batch_size",
    "timed_progress",
    "to_compute_device",
    "to_float32",
    "to_model_inputs",
]


DEFAULT_EMBED_BATCH_SIZE = 32


def resolve_embed_batch_size(default: int = DEFAULT_EMBED_BATCH_SIZE) -> int:
    """Return the configured GPU embed batch size.

    Reads ``VTSEARCH_EMBED_BATCH_SIZE`` from the environment; non-positive
    or unparseable values fall back to *default*.  Subclasses with tighter
    VRAM constraints (e.g. video models with per-clip frame stacks) can
    pass a smaller *default* without touching the env var.
    """
    raw = os.environ.get("VTSEARCH_EMBED_BATCH_SIZE", "").strip()
    if not raw:
        return max(1, default)
    try:
        val = int(raw)
    except ValueError:
        return max(1, default)
    return val if val > 0 else max(1, default)


def media_from_path(file_path: Any, origin: dict | None = None) -> dict:
    """Build a minimal media dict suitable for :meth:`MediaEmbedder.embed_media`.

    Convenience helper for callers that only have a local file path (uploaded
    files, converter outputs, seed data, CLI utilities).  File-based embedders
    read ``media["media_path"]``; service-based embedders can also inspect
    *origin* when supplied.
    """
    from pathlib import Path  # noqa: PLC0415

    p = Path(file_path)
    return {
        "media_path": str(p.resolve()),
        "origin": origin,
        "origin_name": p.name,
        "filename": p.name,
        "custom_metadata": None,
    }


#: Key under which an embedder instance stashes its :class:`_ProgressSlot`.
_PROGRESS_SLOT_KEY = "_progress_slot"


class _ProgressSlot:
    """Per-embedder progress state: a process-wide default + a per-thread override.

    *default* is what :func:`vtscore.media.set_progress_callback` wires in once
    at startup (the host application's progress sink); every thread sees it.
    *local* carries the per-thread override installed by
    :meth:`MediaEmbedder.progress_scope` (or a plain ``emb._on_progress = cb``
    assignment) for the duration of one embed / model-load pass.
    """

    __slots__ = ("default", "local")

    def __init__(self, default: ProgressCallback) -> None:
        self.default = default
        self.local = threading.local()


def _progress_slot(emb: "MediaEmbedder") -> _ProgressSlot:
    """Return *emb*'s :class:`_ProgressSlot`, creating it on first use.

    Created lazily (rather than in ``__init__``) so an embedder subclass that
    never calls ``super().__init__()`` still gets one.  ``dict.setdefault`` is
    atomic under the GIL, so two threads racing to create the slot agree on a
    single winner instead of one silently discarding the other's callback.
    """
    state = vars(emb)
    slot = state.get(_PROGRESS_SLOT_KEY)
    if slot is None:
        slot = state.setdefault(_PROGRESS_SLOT_KEY, _ProgressSlot(_noop_progress))
    return slot


class _ThreadLocalProgress:
    """Data descriptor backing :attr:`MediaEmbedder._on_progress`.

    Embedders are process-wide singletons (``vtscore.media._embedder_registry``),
    so a plain instance attribute made the progress callback shared mutable
    state: two concurrent dataset loads on the same embedder would each assign
    their own tracker callback, and the second assignment re-routed the *first*
    load's still-running ``embed_media_bulk`` into the second load's tracker.
    That mis-drew the progress bar and — because tracker callbacks call
    ``check_cancelled()`` — let cancelling one load raise ``CancelledError``
    inside the other's embed pass, aborting the wrong dataset.  Each pass's
    ``finally`` then restored a callback captured before the other's assignment,
    silencing whichever load was still running.

    Reads and writes are therefore scoped to the calling thread: a write only
    ever redirects the progress of embed / model-load calls made *by that
    thread*, which is exactly what every save-and-restore call site wants.  A
    thread that never assigned anything reads the process-wide default
    (:meth:`MediaEmbedder.set_default_progress_callback`), which the app wires
    to :func:`~vtscore.concurrency.progress.update_progress` — itself a
    per-thread resolution, so an unscoped load on a thread that bound nothing
    lands in a no-op rather than on a channel nobody can terminate.  Background
    warm-ups running *inside* a load that did bind a tracker say they want no
    progress surface with :meth:`MediaEmbedder.silent_progress`.
    """

    def __get__(self, obj: "MediaEmbedder | None", objtype: type | None = None) -> Any:
        if obj is None:
            return self
        slot = _progress_slot(obj)
        cb = getattr(slot.local, "cb", None)
        return slot.default if cb is None else cb

    def __set__(self, obj: "MediaEmbedder", value: ProgressCallback | None) -> None:
        _progress_slot(obj).local.cb = value


class MediaEmbedder(ABC):
    """Abstract base class for media embedders.

    A *media embedder* takes a media file (or a text description) and produces
    a fixed-size vector embedding.  Each embedder is associated with exactly one
    :class:`MediaType` (via :attr:`media_type_id`), but a single media type may
    have multiple embedders (e.g. different CLIP variants for images).

    Subclasses must implement:

    * :attr:`name`: unique human-readable identifier (also used as the
      registry key).
    * :attr:`media_type_id`: which media type this embedder works with.
    * :meth:`load_models`: load (and cache) the embedding model.
    * :meth:`embed_media`: embed a media file from disk.
    * :meth:`embed_text`: embed a text query in the same vector space.
    """

    _model_load_lock: threading.Lock

    # Global lock that serialises all ``embed_media`` calls across every
    # embedder type.
    _embed_lock = threading.Lock()

    #: Progress sink for this embedder's model loads and bulk passes.  Reads
    #: and writes are **per-thread** over a process-wide default; see
    #: :class:`_ThreadLocalProgress` for why, and prefer :meth:`progress_scope`
    #: over assigning it directly.
    _on_progress: ProgressCallback = _ThreadLocalProgress()  # type: ignore[assignment]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._model_load_lock = threading.Lock()

    def set_default_progress_callback(self, callback: ProgressCallback) -> None:
        """Set the process-wide fallback progress sink for this embedder.

        This is the callback every thread sees when it has not installed a
        :meth:`progress_scope` of its own; :func:`vtscore.media.set_progress_callback`
        calls it once at application startup.  Unlike assigning
        :attr:`_on_progress` (which is thread-scoped by design), this is
        deliberately visible across threads.
        """
        _progress_slot(self).default = callback

    @contextlib.contextmanager
    def progress_scope(self, callback: ProgressCallback):
        """Route this embedder's progress to *callback* for the calling thread.

        Restores whatever the calling thread had installed before (usually
        nothing, meaning the process-wide default) on exit, including on
        exception.  Other threads are unaffected for the whole scope, so two
        concurrent dataset loads sharing this singleton embedder each keep
        their own tracker.
        """
        slot = _progress_slot(self)
        prev = getattr(slot.local, "cb", None)
        slot.local.cb = callback
        try:
            yield
        finally:
            slot.local.cb = prev

    def silent_progress(self):
        """Suppress this embedder's progress for the calling thread.

        Sugar over :meth:`progress_scope` for background warm-ups that have no
        progress surface of their own (the smart-preload threads, the
        post-import embedder warm-up).  Without it a warm-up that runs on a
        thread which *did* bind a tracker — the tail of an import, say — would
        narrate itself onto that import's row after the import is over.
        """
        return self.progress_scope(_noop_progress)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this embedder, e.g. ``"clap"``, ``"siglip"``."""

    @property
    def display_name(self) -> str:
        """Human-readable label for this embedder, shown in pickers.

        Defaults to :attr:`name` so legacy embedders keep working unchanged.
        Subclasses should override to surface a friendlier label (e.g.
        ``"SigLIP (general images)"``) while the raw :attr:`name` stays
        available as a secondary line for power users.
        """
        return self.name

    @property
    def model_id(self) -> Optional[str]:
        """Concrete identifier of the pretrained model this embedder loads.

        Usually the HuggingFace Hub repo id of the checkpoint (e.g.
        ``"google/siglip-base-patch16-224"``).  Unlike :attr:`name` (a VTSearch
        slug) and :attr:`display_name` (a friendly label), this is the *exact*
        model a third party would download to reproduce the embedding space, so
        the portable-detector export surfaces it in the bundle manifest/README to
        make the bundle fully actionable.

        A direct weights URL is acceptable where there is no plain repo id (e.g.
        EUPE, loaded from a ``.pt`` URL via ``torch.hub``).  ``None`` (the
        default) means the embedder has no single downloadable model id worth
        surfacing — the classical SIFT/VLAD structural embedder, or FaceNet whose
        weights ship inside ``facenet-pytorch``.
        """
        return None

    @property
    def embedding_dim(self) -> Optional[int]:
        """Output dimensionality of the vectors this embedder produces.

        Descriptor metadata, not derived from a loaded model: it lets tooling
        (the generated docs inventories, UI hints) report the dimension without
        downloading weights.  ``None`` means the dimension is unknown or
        variable; built-in embedders all declare a concrete value.
        """
        return None

    @property
    @abstractmethod
    def media_type_id(self) -> str:
        """The ``type_id`` of the media type this embedder works with."""

    @property
    def is_default(self) -> bool:
        """Whether this embedder is the default for its media type.

        Exactly one embedder per media type should override this to ``True``.
        :func:`vtscore.media.embedders_for_type` returns defaults first so
        callers using ``embedders_for_type(t)[0]`` get the default.
        """
        return False

    @property
    def eval_only(self) -> bool:
        """Whether this embedder exists for measurement rather than for users.

        An eval-only embedder is a *research arm*: it is registered, resolvable
        by name (:func:`vtscore.media.get_embedder`), and usable by the eval
        harness and the pre-embedded pile, but it is withheld from every
        app-facing listing -- the pickers, the per-media-type default, and the
        serialised inventory the frontend reads.

        The distinction is not cosmetic.  A study arm is chosen because it
        *differs* from the shipped embedder in one controlled way; nothing in
        that choice says it is good, supported, or licensed for the app.  A
        deployment can already hide a plugin (``hidden_plugins``), but that is a
        *setting* someone has to apply -- this is a property of the code, so an
        eval arm cannot reach a picker by a deployment forgetting.

        Resolution by name stays open on purpose: a pile cell embedded with an
        eval-only embedder must still load, or the study could not read its own
        vectors back.

        Deliberately **not** in :meth:`to_dict`. The one serialised listing,
        :func:`vtscore.media.all_embedders_dict`, filters eval-only embedders
        out, so the field could only ever serialise as ``False`` -- a constant
        in the API contract, and eighteen exact-equality assertions to carry it.
        Ask the embedder, not its dict.
        """
        return False

    @property
    def supports_text(self) -> bool:
        """Whether this embedder can embed text queries into the same vector space.

        Cross-modal embedders (CLIP, SigLIP, CLAP, X-CLIP) return ``True`` so
        features like text search and description-enrichment are offered.
        Vision-only or patch-based encoders (DINOv3, EUPE) return ``False`` -
        :meth:`embed_text` will not produce meaningful vectors and the UI
        should hide text-search affordances for datasets using them.
        """
        return True

    @property
    def supports_patch_regions(self) -> bool:
        """Whether this embedder produces patch-level vectors and a region tree.

        Patch-based image encoders (DINOv2, DINOv3, EUPE) return ``True``; the
        dataset loader then asks them for a :class:`PatchEmbedOutput` per image
        and stores a hierarchical region set plus the raw patch grid alongside
        the usual ``media["embeddings"]`` vector.  Single-vector embedders return
        ``False`` and the patch-region pipeline is skipped entirely.
        """
        return False

    @property
    def supports_geometric_verification(self) -> bool:
        """Whether this embedder produces local features for instance matching.

        Structural embedders (SIFT/VLAD, and learned-local-feature variants
        later) return ``True``; the dataset loader then asks them for a
        :class:`~vtscore.media.structural.StructuralFeatures` per image and
        stores it as ``media["local_features"]`` alongside the VLAD
        ``media["embeddings"]`` vector, enabling the geometric re-rank + match-stat
        verification paths.  All other embedders return ``False`` and the
        structural pipeline is skipped entirely.

        The flag is deliberately media-agnostic (not ``supports_*_image_*``)
        so an audio constellation-fingerprint backend can reuse it without an
        interface change.
        """
        return False

    @property
    def embed_batch_size(self) -> int:
        """How many items to forward through the model in one GPU call.

        Default reads :envvar:`VTSEARCH_EMBED_BATCH_SIZE` (falling back to
        :data:`DEFAULT_EMBED_BATCH_SIZE`).  Subclasses with tighter VRAM
        budgets (typically video models that stack frames per clip)
        should override to pass a smaller default to
        :func:`resolve_embed_batch_size`.
        """
        return resolve_embed_batch_size()

    @property
    def license_notice(self) -> Optional[str]:
        """User-facing licence warning shown before a user selects this embedder.

        ``None`` (the default) means the embedder has no special licensing
        constraints worth surfacing.  Embedders distributed under a research-
        only or otherwise-restrictive licence (e.g. facebookresearch/EUPE under
        the FAIR Noncommercial Research Licence) return a short human-readable
        string the UI shows on the embedder picker so users know before they
        produce any outputs.  This is advisory; there is no acceptance
        click; users who object pick a different embedder.
        """
        return None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load (and cache) the embedding model.

        Called lazily the first time this embedder needs to produce a vector.
        Implementations must be idempotent; a second call should be a no-op.

        Subclasses should override :meth:`_load_models_impl` (not this method).
        This wrapper catches :class:`ImportError` and re-raises with an
        actionable message so that missing dependencies surface clearly.

        A per-class lock serialises concurrent callers so that only one
        thread performs the actual load; others wait and then return
        immediately (the subclass ``_load_models_impl`` checks
        ``self._model is not None``).

        A load whose caller installed no :meth:`progress_scope` reports through
        the process-wide default sink, which resolves per-thread and drops the
        ticks when the thread bound no tracker.
        """
        if getattr(self, "_model", None) is not None:
            return
        with self._model_load_lock:
            try:
                self._load_models_impl()
            except ImportError as exc:
                raise ImportError(
                    f"{exc}. Required by the '{self.name}' embedder. "
                    f"Install dependencies with: pip install -e '.[cpu,dev]'"
                ) from exc

    @abstractmethod
    def _load_models_impl(self) -> None:
        """Subclass hook: load the embedding model.

        Override this instead of :meth:`load_models`.
        """

    def models_loaded(self) -> bool:
        """Whether this embedder's model is already resident in this process.

        The supported way to ask "would :meth:`load_models` do any work?"
        without doing it. Callers that merely need vectors should not care;
        this is for code that has to *plan around* the load — pacing a progress
        bar (the text-sort route budgets its ``load_model`` step out of the bar
        entirely when the encoder is already resident, see
        :func:`vtscore.timing.step_weights`) or skipping a speculative warm-up.

        Reads the same ``_model`` attribute convention :meth:`load_models` and
        :meth:`loaded_backbone` rely on, so it works for any embedder built the
        usual way. An embedder that holds its backbone elsewhere should
        **override this** alongside :meth:`loaded_backbone`; the default's
        answer would otherwise be a permanent ``False``, which is the safe
        direction (a caller re-plans for a load that turns out to be free)
        but wrong.
        """
        return getattr(self, "_model", None) is not None

    def loaded_backbone(self) -> tuple[Any, Any]:
        """Return ``(model, processor)`` for this embedder's underlying backbone.

        This is the *supported* way to reach the raw model behind an
        embedder - for custom forward passes, intermediate-layer probes, and
        the convenience getters in :mod:`vtscore.embedding.loader`
        (:func:`~vtscore.embedding.loader.get_clap_model` and friends).
        Prefer :meth:`embed_media` / :meth:`embed_text` for anything that
        just needs vectors; reach for the backbone only when you genuinely
        need to drive it yourself.

        Loads the model first if it is not already resident, so callers need
        not call :meth:`load_models` themselves.  The second element is
        ``None`` for embedders whose backbone needs no separate processor
        (e.g. a ``SentenceTransformer``).

        The default implementation reads the ``_model`` / ``_processor``
        attribute convention that :meth:`load_models` itself relies on, so it
        works for any embedder built the usual way.  An embedder that holds
        its backbone elsewhere - or wraps several - should **override this**
        rather than leave callers guessing.

        :raises RuntimeError: when no backbone is resident after
            :meth:`load_models` returned.  Failing loudly here beats handing
            back ``None`` and surfacing as an unrelated crash later.
        """
        self.load_models()
        model = getattr(self, "_model", None)
        if model is None:
            raise RuntimeError(
                f"The '{self.name}' embedder exposes no backbone after load_models(). "
                "Override loaded_backbone() to return (model, processor)."
            )
        return model, getattr(self, "_processor", None)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_media(self, media: dict) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for *media*.

        *media* is a media dict (the same shape produced by the dataset
        loader).  File-based embedders pull ``Path(media["media_path"])``;
        service-based embedders can use ``media["origin"]``,
        ``media["origin_name"]``, ``media.get("custom_metadata")`` etc. to
        look the content up remotely without touching disk.

        Acquires :attr:`_embed_lock` so that only one forward pass runs at a
        time across all embedder types.  Subclasses must override
        :meth:`_embed_media_impl` (not this method).

        The returned vector is **L2-normalized** here (via
        :func:`vtscore.embedding.normalize.l2_normalize`) so that every
        embedding stored in ``medias`` is unit-norm regardless of which
        embedder produced it; subclasses must not (and need not) normalize
        themselves.

        Returns ``None`` if the media cannot be embedded.
        """
        from vtscore.embedding.normalize import l2_normalize  # noqa: PLC0415

        with self._embed_lock:
            vec = self._embed_media_impl(media)
        return None if vec is None else l2_normalize(vec)

    @abstractmethod
    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        """Subclass hook: embed a single media item.

        Override this instead of :meth:`embed_media`.
        """

    # ------------------------------------------------------------------
    # Bulk embedding
    # ------------------------------------------------------------------

    def embed_media_bulk(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        """Embed every item in *medias* and return a same-length list of vectors.

        Positions where an item could not be embedded contain ``None``.

        The default implementation dispatches to :meth:`embed_media` per
        item; each call acquires :attr:`_embed_lock` individually so
        concurrent callers can interleave, and emits per-item progress
        via :attr:`_on_progress` so long runs stay visible in the UI.

        Subclasses backed by a service that natively accepts many items
        per request should override :meth:`_embed_media_bulk_impl`.  If
        they chunk internally (batching), they are responsible for
        emitting their own progress updates through :attr:`_on_progress`.

        Every returned vector is **L2-normalized** here so the stored-as-
        unit-norm invariant holds for the bulk path too (the default impl
        already routes through :meth:`embed_media`, so re-normalizing is a
        harmless no-op; overriding impls that batch raw outputs are covered
        here).
        """
        if not medias:
            return []
        from vtscore.embedding.normalize import l2_normalize  # noqa: PLC0415

        vectors = self._embed_media_bulk_impl(medias)
        return [None if v is None else l2_normalize(v) for v in vectors]

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        """Subclass hook: embed a list of media items.

        Default: loop over :meth:`embed_media`, emitting per-item progress.
        Override to replace the per-item loop with a single bulk request,
        or to batch internally in chunks sized for a remote API.
        """
        total = len(medias)
        results: list[Optional[np.ndarray]] = []
        for i, m in enumerate(medias):
            self._on_progress("embedding", "Embedding...", i + 1, total)
            results.append(self.embed_media(m))
        return results

    def embed_medias(self, medias: dict[int, dict]) -> dict[int, Optional[np.ndarray]]:
        """Bulk-embed an id→media dict; return id→vector (or ``None``) dict.

        Convenience wrapper around :meth:`embed_media_bulk` for callers that
        already have medias keyed by ID, typically dataset importers that
        have built the medias dict before embedding.  IDs whose embedding
        failed map to ``None`` in the returned dict, mirroring the position-
        based ``None`` contract of :meth:`embed_media_bulk`.
        """
        if not medias:
            return {}
        keys = list(medias.keys())
        values = [medias[k] for k in keys]
        vectors = self.embed_media_bulk(values)
        return dict(zip(keys, vectors))

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Return an embedding of *text* in the **same vector space** as :meth:`embed_media`.

        The result is **L2-normalized** here so query vectors are unit-norm
        just like stored media embeddings; this is what lets
        :mod:`vtscore.training.region_similarity` score with a plain dot
        product instead of re-normalizing on every comparison.  Subclasses
        override :meth:`_embed_text_impl` (not this method) and need not
        normalize themselves.

        Returns ``None`` when this embedder cannot embed text.
        """
        vec = self._embed_text_impl(text)
        if vec is None:
            return None
        from vtscore.embedding.normalize import l2_normalize  # noqa: PLC0415

        return l2_normalize(vec)

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        """Subclass hook: embed a text query.

        Override this instead of :meth:`embed_text`.  The default returns
        ``None`` (text sorting unavailable).
        """
        return None

    def patch_forward(self, media: dict) -> Optional["PatchEmbedOutput"]:  # noqa: F821
        """Return per-patch features for one image.

        Patch-based image encoders (DINOv2, DINOv3, EUPE) override this to
        return a :class:`~vtscore.media.patch_embed.PatchEmbedOutput`
        carrying the CLS vector, the per-patch grid, and a per-patch saliency
        map.  Single-vector embedders leave the default in place and the
        loader pipeline skips the patch-region step for their datasets.

        The dataset loader gates calls on :attr:`supports_patch_regions`:
        if you set that flag ``True``, you must override this method.

        Acquires :attr:`_embed_lock` so the patch forward pass interleaves
        with single-vector embedders' forward passes on the same lock.
        Subclasses override :meth:`_patch_forward_impl` (not this method).

        Returns ``None`` if the media can't be loaded.
        """
        from vtscore.media.patch_embed import PatchEmbedOutput  # noqa: F401, PLC0415

        with self._embed_lock:
            return self._patch_forward_impl(media)

    def _patch_forward_impl(self, media: dict) -> Optional["PatchEmbedOutput"]:  # noqa: F821
        """Subclass hook for :meth:`patch_forward`.

        Default returns ``None``.  Patch-capable embedders override this.
        """
        return None

    def patch_forward_bulk(self, medias: list[dict]) -> list[Optional["PatchEmbedOutput"]]:  # noqa: F821
        """Return per-patch features for every image in *medias*.

        Patch-capable embedders override :meth:`_patch_forward_bulk_impl`
        to batch the forward pass through their backbone. The default
        loops :meth:`patch_forward` per item and emits per-item progress
        via :attr:`_on_progress`, matching the contract of
        :meth:`embed_media_bulk`.

        Positions where patch-forward returned ``None`` (failed decode,
        unsupported, etc.) contain ``None``.
        """
        if not medias:
            return []
        return self._patch_forward_bulk_impl(medias)

    def _patch_forward_bulk_impl(self, medias: list[dict]) -> list[Optional["PatchEmbedOutput"]]:  # noqa: F821
        """Subclass hook: bulk patch-forward.

        Default: loop over :meth:`patch_forward`, emitting per-item
        progress.  Override to fuse the per-image forward into a single
        batched GPU call.
        """
        total = len(medias)
        results: list[Optional["PatchEmbedOutput"]] = []  # noqa: F821
        for i, m in enumerate(medias):
            self._on_progress("embedding", "Patch-embedding...", i + 1, total)
            results.append(self.patch_forward(m))
        return results

    def local_features_forward(self, media: dict) -> Optional["StructuralFeatures"]:  # noqa: F821
        """Return local instance-matching features for one image.

        Structural embedders (SIFT/VLAD, and learned-local-feature variants
        later) override this to return a
        :class:`~vtscore.media.structural.StructuralFeatures` carrying the
        per-image keypoints and descriptors used by the geometric re-rank and
        the match-statistic verification classifier.  All other embedders
        leave the default in place and the loader skips the structural pass.

        The dataset loader gates calls on
        :attr:`supports_geometric_verification`: if you set that flag
        ``True``, you must override this method.

        Acquires :attr:`_embed_lock` so the feature-detection pass interleaves
        with other embedders' forward passes on the same lock.  Subclasses
        override :meth:`_local_features_forward_impl` (not this method).

        Returns ``None`` if the media can't be loaded.
        """
        with self._embed_lock:
            return self._local_features_forward_impl(media)

    def _local_features_forward_impl(self, media: dict) -> Optional["StructuralFeatures"]:  # noqa: F821
        """Subclass hook for :meth:`local_features_forward`.

        Default returns ``None``.  Structural embedders override this.
        """
        return None

    def local_features_forward_bulk(self, medias: list[dict]) -> list[Optional["StructuralFeatures"]]:  # noqa: F821
        """Return local features for every image in *medias*.

        Structural embedders override :meth:`_local_features_forward_bulk_impl`
        to batch the feature-detection pass.  The default loops
        :meth:`local_features_forward` per item and emits per-item progress
        via :attr:`_on_progress`, matching the contract of
        :meth:`patch_forward_bulk`.

        Positions where feature detection returned ``None`` (failed decode,
        unsupported, etc.) contain ``None``.
        """
        if not medias:
            return []
        return self._local_features_forward_bulk_impl(medias)

    def _local_features_forward_bulk_impl(self, medias: list[dict]) -> list[Optional["StructuralFeatures"]]:  # noqa: F821
        """Subclass hook: bulk local-feature detection.

        Default: loop over :meth:`local_features_forward`, emitting per-item
        progress.  Override to fuse the per-image detection into a batched call.
        """
        total = len(medias)
        results: list[Optional["StructuralFeatures"]] = []  # noqa: F821
        for i, m in enumerate(medias):
            self._on_progress("embedding", "Detecting features...", i + 1, total)
            results.append(self.local_features_forward(m))
        return results

    @property
    def description_wrappers(self) -> list[str]:
        """Wrapper templates for enriching sort descriptions.

        Each template is a format string containing ``{text}``.  Override in
        subclasses to provide media-specific wrappers that improve embedding
        quality.

        Whether wrappers help is a property of the **embedder**, not of the
        media type: #3127 measured enrichment on/off across 22 eval datasets
        and 560 paired queries and found the ensemble a clear loss on ``e5``,
        ``bge``, ``siglip`` and ``clap``, and a gain only on ``clap_general``
        and ``xclip``.  So an empty list here is a real answer -- "no wrapper
        beats the typed query on this model" -- and not merely an unfilled
        slot.  Return ``[]`` (the default) unless you have measured otherwise;
        :meth:`embed_text_enriched` then degrades to :meth:`embed_text` and the
        Enrich Sort Descriptions setting is a no-op for that embedder.
        """
        return []

    def embed_text_enriched(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* using the average over all description wrappers.

        Falls back to :meth:`embed_text` if no wrappers are defined or all fail.
        """
        wrappers = self.description_wrappers
        if not wrappers:
            return self.embed_text(text)

        embeddings = []
        for wrapper in wrappers:
            wrapped = wrapper.format(text=text)
            vec = self.embed_text(wrapped)
            if vec is not None:
                embeddings.append(vec)

        if not embeddings:
            return self.embed_text(text)

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary of this embedder."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "model_id": self.model_id,
            "media_type_id": self.media_type_id,
            "is_default": self.is_default,
            "supports_text": self.supports_text,
            "supports_patch_regions": self.supports_patch_regions,
            "supports_geometric_verification": self.supports_geometric_verification,
            "license_notice": self.license_notice,
        }
