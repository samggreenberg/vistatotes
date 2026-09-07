"""Structured logging with contextual VTSearch request fields.

Every ``LogRecord`` is tagged with the user, ``dataset_id``, ``detector_id``,
and ``request_id`` active when the record was created. Inside a Flask
request handler the values come from ``flask.g`` (populated by the
``before_request`` middleware in :mod:`app`); inside a background thread
they come from the thread-locals already maintained by
:mod:`vtsearch.auth` and :mod:`vtscore.state.core`
(``thread_user`` / ``thread_dataset_context`` / ``thread_detector_context``
context managers, or the bare ``set_thread_*`` setters they wrap).

Two output formats are supported, selected by the
``VTSEARCH_LOG_FORMAT`` env var:

* ``json`` (default): one JSON object per line, suitable for log aggregators.
* ``text``: human-readable bracketed-tag form, suitable for local dev.

The level is controlled by ``VTSEARCH_LOG_LEVEL`` (default ``WARNING``).

Existing ``logging.getLogger(__name__)`` call sites scattered throughout
the codebase inherit the context automatically; no per-call ``extra=``
dict is needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from typing import Any, TextIO

_DEFAULT_LEVEL = "WARNING"
_DEFAULT_FORMAT = "json"

# Order matters: the JsonFormatter emits these in this order, and the
# TextFormatter walks them in this order to build the bracketed tag list.
_CONTEXT_FIELDS: tuple[str, ...] = ("request_id", "user", "dataset_id", "detector_id")

# Built-in LogRecord attributes, used to filter the record's __dict__
# down to caller-supplied ``extra=`` keys when emitting JSON.
_STD_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


def _resolve_context() -> dict[str, Any]:  # noqa: C901
    """Pull the active VTSearch context from Flask ``g`` or thread-locals.

    Returns a dict with whatever subset of (``request_id``, ``user``,
    ``dataset_id``, ``detector_id``) is available; missing values are
    omitted (not set to ``None``) so the JSON output stays compact.

    Defensive: any exception while resolving is swallowed so a misconfigured
    Flask app or a half-initialised thread context can never take down
    logging. Logging that crashes is worse than logging that's missing a
    field.
    """
    out: dict[str, Any] = {}

    try:
        from flask import g, has_request_context  # local import; Flask may not be loaded yet
    except Exception:
        has_request_context = lambda: False  # noqa: E731
        g = None  # type: ignore[assignment]

    in_request = False
    try:
        in_request = bool(has_request_context())
    except Exception:
        in_request = False

    ds_ctx = None
    det_ctx = None
    if in_request and g is not None:
        try:
            rid = getattr(g, "request_id", None)
            if rid:
                out["request_id"] = rid
            user = getattr(g, "user", None)
            if user:
                out["user"] = user
            ds_ctx = getattr(g, "_dataset_context", None)
            det_ctx = getattr(g, "_detector_context", None)
        except Exception:
            pass

    # Thread-local fallbacks (background jobs propagate these via the
    # thread_user / thread_dataset_context / thread_detector_context
    # context managers in vtsearch.auth and vtscore.state.core).
    if "user" not in out:
        try:
            from vtsearch.auth import get_thread_user

            tl_user = get_thread_user()
            if tl_user:
                out["user"] = tl_user
        except Exception:
            pass

    if ds_ctx is None or det_ctx is None:
        try:
            from vtscore.state.core import (
                get_thread_dataset_context,
                get_thread_detector_context,
            )

            if ds_ctx is None:
                ds_ctx = get_thread_dataset_context()
            if det_ctx is None:
                det_ctx = get_thread_detector_context()
        except Exception:
            pass

    ds_id = getattr(ds_ctx, "dataset_id", None) if ds_ctx is not None else None
    if ds_id:
        out["dataset_id"] = ds_id
    det_id = getattr(det_ctx, "detector_id", None) if det_ctx is not None else None
    if det_id:
        out["detector_id"] = det_id

    return out


class ContextFilter(logging.Filter):
    """Attach VTSearch context fields to every ``LogRecord``.

    Installed on the root logger's stream handler so every record routed
    through it gets the fields, regardless of which module produced it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _resolve_context()
        for field in _CONTEXT_FIELDS:
            # Don't overwrite an explicit ``extra={"dataset_id": "..."}``
            # value passed by the caller (they know what they're doing).
            if not hasattr(record, field):
                setattr(record, field, ctx.get(field))
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per ``LogRecord``, on a single line."""

    def format(self, record: logging.LogRecord) -> str:
        ts_struct = time.gmtime(record.created)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", ts_struct) + f".{int(record.msecs):03d}Z"
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        # Caller-supplied extras (``logger.info("...", extra={"job_id": x})``).
        for key, value in record.__dict__.items():
            if key in _STD_RECORD_ATTRS or key in payload or key.startswith("_"):
                continue
            if key in _CONTEXT_FIELDS:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Plain-text formatter with bracketed context tags.

    Example::

        WARNING vtsearch.routes.datasets [req=abc123 ds=my-dataset]: Foo
    """

    _SHORT = {
        "request_id": "req",
        "user": "user",
        "dataset_id": "ds",
        "detector_id": "det",
    }

    def format(self, record: logging.LogRecord) -> str:
        tags: list[str] = []
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if not value:
                continue
            short_val = str(value)[:8] if field == "request_id" else str(value)
            tags.append(f"{self._SHORT[field]}={short_val}")
        ctx = f" [{' '.join(tags)}]" if tags else ""
        base = f"{record.levelname} {record.name}{ctx}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


_VOCAB_TOKEN_WARN_RE = re.compile(r"(bos|eos|pad)_token_id must be `None` or an integer within the vocabulary")


class _TransformersVocabTokenFilter(logging.Filter):
    """Drop transformers' bos/eos/pad token-out-of-vocab warnings.

    CLIP-derived models (CLAP, X-CLIP, plain CLIP) carry CLIP's 49406/49407
    BOS/EOS tokens against a 32k sentencepiece vocab in a sibling text
    sub-config. transformers logs a config-validation warning on every load;
    the mismatch is harmless and there's nothing for the user to fix.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("transformers"):
            return True
        try:
            return _VOCAB_TOKEN_WARN_RE.search(record.getMessage()) is None
        except Exception:
            return True


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure root logging for VTSearch.

    Resolution order for each parameter:

    1. Explicit keyword argument.
    2. Environment variable (``VTSEARCH_LOG_LEVEL`` / ``VTSEARCH_LOG_FORMAT``).
    3. Built-in default (``WARNING`` / ``json``).

    Idempotent: safe to call multiple times. Replaces any handlers added
    by a previous call (or by an earlier ``logging.basicConfig``) so log
    lines aren't emitted twice.
    """
    level_str = (level or os.environ.get("VTSEARCH_LOG_LEVEL") or _DEFAULT_LEVEL).upper()
    level_value = getattr(logging, level_str, logging.WARNING)

    fmt_kind = (fmt or os.environ.get("VTSEARCH_LOG_FORMAT") or _DEFAULT_FORMAT).lower()
    formatter: logging.Formatter = JsonFormatter() if fmt_kind == "json" else TextFormatter()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())
    handler.addFilter(_TransformersVocabTokenFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level_value)

    # Quiet down chatty libraries; preserved from app.py's old basicConfig.
    #
    # ``huggingface_hub`` stays pinned to ERROR regardless of level: its
    # INFO/DEBUG output is download-progress chatter nobody asked for.
    #
    # ``werkzeug`` is different. Its INFO records *are* the dev-server access
    # log (one ``"GET /api/... 200"`` line per request), so we only silence it
    # at the default WARNING+ levels and let it through once the operator opts
    # into INFO/DEBUG (``python app.py -v`` or ``VTSEARCH_LOG_LEVEL=info``).
    # Pinning it to ERROR unconditionally was the ``no-access-log`` bug: there
    # was no way to see request activity at all, even with the level turned up.
    werkzeug_level = level_value if level_value <= logging.INFO else logging.ERROR
    logging.getLogger("werkzeug").setLevel(werkzeug_level)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)


def install_transformers_logging_bridge() -> None:
    """Route the ``transformers`` library's logs through our root handler.

    transformers configures its own stderr handler with a ``[transformers]``
    prefix formatter and sets ``propagate=False`` on its library root logger,
    so records from ``transformers.*`` never reach the handler installed by
    :func:`setup_logging`. That makes our context tags and our
    :class:`_TransformersVocabTokenFilter` ineffective for transformers' own
    output. This bridge disables their default handler and re-enables
    propagation so transformers records flow through our formatter and
    filters like every other library's.

    Deferred from ``setup_logging`` because importing ``transformers.utils.logging``
    pulls in the full ``transformers`` package, which we don't want to pay for
    in unit tests that stub embedders. Call once during app startup before any
    model load; :func:`vtscore.embedding.loader.initialize_models` is the
    canonical call site.

    That import costs ~0.7s with the venv's metadata in page cache, but it is
    **not** bounded by that: transformers builds
    ``importlib.metadata.packages_distributions()`` at module import, which
    stats every file recorded by every installed distribution (~85k of them
    here). On an NFS venv with a cold dentry cache that took 16 minutes of
    silent startup on the GRID (issue #3715), which is why this call seeds the
    stat-free replacement first. The seed is idempotent, so paying for it here
    as well as in ``initialize_models`` costs nothing and covers callers that
    reach this bridge by another route.
    """
    # Imported lazily: ``logging_config`` is app.py's very first import, and a
    # module-level ``vtscore`` import here would pull the library package in
    # before logging is configured.
    from vtscore.utils.import_metadata import seed_packages_distributions  # noqa: PLC0415

    seed_packages_distributions()
    try:
        import transformers.utils.logging as hf_logging  # noqa: PLC0415
    except Exception:
        return
    try:
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
    except Exception:
        pass


def new_request_id() -> str:
    """Generate a short request id (12 hex chars from a uuid4)."""
    return uuid.uuid4().hex[:12]
