"""Tests for structured logging + request-id middleware."""

from __future__ import annotations

import io
import json
import logging
import threading

from vtsearch.logging_config import (
    ContextFilter,
    JsonFormatter,
    TextFormatter,
    install_transformers_logging_bridge,
    new_request_id,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_handler(stream: io.StringIO, fmt: str = "json") -> logging.Handler:
    """Build a handler with the same wiring setup_logging() uses, but
    pointed at an in-memory stream so the test can read what was written."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.addFilter(ContextFilter())
    return handler


def _emit(stream: io.StringIO, fmt: str, fn) -> list[dict]:
    """Install a fresh handler, run ``fn`` (which calls a logger), then
    return the parsed JSON lines (or raw text lines if fmt == 'text')."""
    root = logging.getLogger()
    prior_level = root.level
    prior_handlers = list(root.handlers)
    for h in prior_handlers:
        root.removeHandler(h)
    handler = _build_handler(stream, fmt=fmt)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        fn()
    finally:
        root.removeHandler(handler)
        for h in prior_handlers:
            root.addHandler(h)
        root.setLevel(prior_level)
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    if fmt == "json":
        return [json.loads(line) for line in lines]
    return lines  # type: ignore[return-value]


def _emit_with_empty_context(stream: io.StringIO, fmt: str, fn):
    """Like ``_emit`` but clears the test-default dataset/detector thread-locals
    for the duration of the call. The autouse ``reset_state`` fixture installs
    ``_test_default`` contexts; tests that assert *absence* of context need
    those out of the way."""
    from vtsearch.auth import set_thread_user
    from vtscore.state.core import (
        get_thread_dataset_context,
        get_thread_detector_context,
        set_thread_dataset_context,
        set_thread_detector_context,
    )

    prior_ds = get_thread_dataset_context()
    prior_det = get_thread_detector_context()
    set_thread_dataset_context(None)
    set_thread_detector_context(None)
    set_thread_user(None)
    try:
        return _emit(stream, fmt, fn)
    finally:
        set_thread_dataset_context(prior_ds)
        set_thread_detector_context(prior_det)


# ---------------------------------------------------------------------------
# Formatter unit tests
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def test_basic_fields(self):
        stream = io.StringIO()
        records = _emit(stream, "json", lambda: logging.getLogger("vtsearch.test").warning("hello %s", "world"))
        assert len(records) == 1
        rec = records[0]
        assert rec["level"] == "WARNING"
        assert rec["logger"] == "vtsearch.test"
        assert rec["msg"] == "hello world"
        assert "ts" in rec and rec["ts"].endswith("Z")

    def test_no_context_outside_request(self):
        """Context fields should be absent (not null) when no context exists."""
        stream = io.StringIO()
        records = _emit_with_empty_context(stream, "json", lambda: logging.getLogger("vtsearch.test").info("no ctx"))
        assert "request_id" not in records[0]
        assert "dataset_id" not in records[0]
        assert "detector_id" not in records[0]
        assert "user" not in records[0]

    def test_extra_fields_included(self):
        stream = io.StringIO()
        records = _emit(
            stream,
            "json",
            lambda: logging.getLogger("vtsearch.test").info("job done", extra={"job_id": "abc", "count": 3}),
        )
        assert records[0]["job_id"] == "abc"
        assert records[0]["count"] == 3

    def test_non_json_extras_repr_fallback(self):
        class Weird:
            def __repr__(self):
                return "<weird>"

        stream = io.StringIO()
        records = _emit(
            stream,
            "json",
            lambda: logging.getLogger("vtsearch.test").info("x", extra={"obj": Weird()}),
        )
        assert records[0]["obj"] == "<weird>"

    def test_exception_included(self):
        stream = io.StringIO()

        def go():
            try:
                raise ValueError("boom")
            except ValueError:
                logging.getLogger("vtsearch.test").exception("caught")

        records = _emit(stream, "json", go)
        assert records[0]["msg"] == "caught"
        assert "ValueError: boom" in records[0]["exc"]


class TestTextFormatter:
    def test_basic_line_no_context(self):
        stream = io.StringIO()
        lines = _emit_with_empty_context(stream, "text", lambda: logging.getLogger("vtsearch.test").warning("hi"))
        assert lines == ["WARNING vtsearch.test: hi"]

    def test_basic_line_with_thread_local_context(self):
        """When thread-local context is set, the bracketed tags appear."""
        from vtscore.state.core import (
            DatasetContext,
            get_thread_dataset_context,
            get_thread_detector_context,
            set_thread_dataset_context,
            set_thread_detector_context,
        )

        prior_ds = get_thread_dataset_context()
        prior_det = get_thread_detector_context()
        try:
            set_thread_dataset_context(DatasetContext("my-ds"))
            set_thread_detector_context(None)
            stream = io.StringIO()
            lines = _emit(stream, "text", lambda: logging.getLogger("vtsearch.test").warning("hi"))
            assert lines == ["WARNING vtsearch.test [ds=my-ds]: hi"]
        finally:
            set_thread_dataset_context(prior_ds)
            set_thread_detector_context(prior_det)


# ---------------------------------------------------------------------------
# Context resolution
# ---------------------------------------------------------------------------


class TestRequestContext:
    """When inside a Flask request, log records inherit g.* fields."""

    def test_request_id_and_user_on_record(self):
        """A log line emitted inside a Flask request context carries
        request_id and user pulled from ``g``."""
        import app as app_module
        from flask import g

        stream = io.StringIO()
        with app_module.app.test_request_context("/api/auth/status"):
            g.request_id = new_request_id()
            g.user = "alice"
            expected_rid = g.request_id
            records = _emit(stream, "json", lambda: logging.getLogger("vtsearch.test.probe").warning("inside request"))

        assert len(records) == 1
        rec = records[0]
        assert rec["request_id"] == expected_rid
        assert len(rec["request_id"]) == 12
        assert rec["user"] == "alice"

    def test_response_carries_x_request_id_header(self, client):
        resp = client.get("/api/auth/status")
        assert resp.status_code == 200
        rid = resp.headers.get("X-Request-Id")
        assert rid is not None
        assert len(rid) >= 8

    def test_inbound_request_id_is_honoured(self, client):
        resp = client.get("/api/auth/status", headers={"X-Request-Id": "trace-xyz-123"})
        assert resp.headers["X-Request-Id"] == "trace-xyz-123"

    def test_inbound_request_id_is_truncated(self, client):
        """A malicious caller can't blow up log lines by sending a 10 MB header."""
        huge = "z" * 5000
        resp = client.get("/api/auth/status", headers={"X-Request-Id": huge})
        # Bounded to <= 64 chars.
        assert len(resp.headers["X-Request-Id"]) <= 64
        assert resp.headers["X-Request-Id"] == huge[:64]

    def test_unknown_api_route_still_gets_request_id_header(self, client):
        """Even 404s should carry X-Request-Id so clients can quote it in bug reports."""
        resp = client.get("/api/no-such-endpoint-asdf")
        assert resp.status_code == 404
        assert resp.headers.get("X-Request-Id") is not None


class TestThreadLocalContext:
    """In a background thread, records inherit set_thread_* values."""

    def test_thread_local_user_on_record(self):
        from vtsearch.auth import set_thread_user

        stream = io.StringIO()
        captured: list[dict] = []
        ready = threading.Event()

        def target():
            set_thread_user("alice")
            try:
                handler = _build_handler(stream, fmt="json")
                root = logging.getLogger()
                prior = list(root.handlers)
                for h in prior:
                    root.removeHandler(h)
                root.addHandler(handler)
                root.setLevel(logging.DEBUG)
                try:
                    logging.getLogger("vtsearch.test.bg").info("from-thread")
                finally:
                    root.removeHandler(handler)
                    for h in prior:
                        root.addHandler(h)
                lines = [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]
                captured.extend(lines)
            finally:
                set_thread_user(None)
                ready.set()

        t = threading.Thread(target=target)
        t.start()
        ready.wait(timeout=5)
        t.join(timeout=5)

        assert len(captured) == 1
        assert captured[0]["user"] == "alice"

    def test_thread_local_dataset_and_detector_id(self):
        from vtscore.state.core import (
            DatasetContext,
            DetectorContext,
            set_thread_dataset_context,
            set_thread_detector_context,
        )

        stream = io.StringIO()
        captured: list[dict] = []
        ready = threading.Event()

        def target():
            ds = DatasetContext("my-dataset-42")
            det = DetectorContext("my-detector-7")
            set_thread_dataset_context(ds)
            set_thread_detector_context(det)
            try:
                handler = _build_handler(stream, fmt="json")
                root = logging.getLogger()
                prior = list(root.handlers)
                for h in prior:
                    root.removeHandler(h)
                root.addHandler(handler)
                root.setLevel(logging.DEBUG)
                try:
                    logging.getLogger("vtsearch.test.bg").info("with-ctx")
                finally:
                    root.removeHandler(handler)
                    for h in prior:
                        root.addHandler(h)
                lines = [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]
                captured.extend(lines)
            finally:
                set_thread_dataset_context(None)
                set_thread_detector_context(None)
                ready.set()

        t = threading.Thread(target=target)
        t.start()
        ready.wait(timeout=5)
        t.join(timeout=5)

        assert len(captured) == 1
        assert captured[0]["dataset_id"] == "my-dataset-42"
        assert captured[0]["detector_id"] == "my-detector-7"


# ---------------------------------------------------------------------------
# setup_logging idempotency
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_idempotent_no_duplicate_handlers(self):
        """Calling setup_logging twice should not double up log output."""
        try:
            setup_logging(level="DEBUG", fmt="json")
            count_first = len(logging.getLogger().handlers)
            setup_logging(level="DEBUG", fmt="json")
            count_second = len(logging.getLogger().handlers)
            assert count_first == count_second == 1
        finally:
            # Restore defaults so subsequent tests use the env-driven config.
            setup_logging()

    def test_format_env_var_selects_text(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_LOG_FORMAT", "text")
        try:
            setup_logging()
            handler = logging.getLogger().handlers[0]
            assert isinstance(handler.formatter, TextFormatter)
        finally:
            monkeypatch.delenv("VTSEARCH_LOG_FORMAT", raising=False)
            setup_logging()

    def test_unknown_level_falls_back_to_warning(self):
        try:
            setup_logging(level="NOSUCHLEVEL")
            assert logging.getLogger().level == logging.WARNING
        finally:
            setup_logging()


class TestWerkzeugAccessLogLevel:
    """werkzeug's INFO records are the dev-server access log. setup_logging
    keeps them quiet by default (the ``no-access-log`` fix) but lets them
    through once the level is raised to INFO/DEBUG, while huggingface_hub
    stays pinned to ERROR regardless."""

    def test_werkzeug_silenced_at_default_warning(self):
        try:
            setup_logging(level="WARNING")
            assert logging.getLogger("werkzeug").level == logging.ERROR
        finally:
            setup_logging()

    def test_werkzeug_follows_info_level(self):
        try:
            setup_logging(level="INFO")
            assert logging.getLogger("werkzeug").level == logging.INFO
        finally:
            setup_logging()

    def test_werkzeug_follows_debug_level(self):
        try:
            setup_logging(level="DEBUG")
            assert logging.getLogger("werkzeug").level == logging.DEBUG
        finally:
            setup_logging()

    def test_huggingface_hub_stays_quiet_even_at_debug(self):
        try:
            setup_logging(level="DEBUG")
            assert logging.getLogger("huggingface_hub").level == logging.ERROR
        finally:
            setup_logging()


class TestTransformersVocabTokenFilter:
    """Filter installed by setup_logging() drops only the bos/eos/pad
    vocab-range warning from transformers; everything else passes through."""

    def test_drops_bos_token_warning(self):
        stream = io.StringIO()
        try:
            setup_logging(level="DEBUG", fmt="text", stream=stream)
            logging.getLogger("transformers.configuration_utils").warning(
                "Model config: bos_token_id must be `None` or an integer within "
                "the vocabulary (between 0 and 31999), got 49406."
            )
            assert stream.getvalue() == ""
        finally:
            setup_logging()

    def test_drops_eos_token_warning_on_descendant_logger(self):
        stream = io.StringIO()
        try:
            setup_logging(level="DEBUG", fmt="text", stream=stream)
            logging.getLogger("transformers.models.clap.configuration_clap").warning(
                "Model config: eos_token_id must be `None` or an integer within "
                "the vocabulary (between 0 and 31999), got 49407."
            )
            assert stream.getvalue() == ""
        finally:
            setup_logging()

    def test_unrelated_transformers_warning_passes_through(self):
        stream = io.StringIO()
        try:
            setup_logging(level="DEBUG", fmt="text", stream=stream)
            logging.getLogger("transformers").warning("some other thing happened")
            assert "some other thing happened" in stream.getvalue()
        finally:
            setup_logging()

    def test_non_transformers_logger_with_matching_text_passes_through(self):
        """Filter is scoped to transformers loggers so it can't accidentally
        eat a same-shaped message logged from elsewhere."""
        stream = io.StringIO()
        try:
            setup_logging(level="DEBUG", fmt="text", stream=stream)
            logging.getLogger("vtsearch.test").warning(
                "bos_token_id must be `None` or an integer within the vocabulary..."
            )
            assert "bos_token_id" in stream.getvalue()
        finally:
            setup_logging()


class TestTransformersLoggingBridge:
    """The bridge disables transformers' default handler and re-enables
    propagation so its records flow through our root handler (and our
    vocab-token filter)."""

    def test_bridge_disables_default_handler_and_enables_propagation(self):
        import transformers.utils.logging as hf_logging

        hf_logging._reset_library_root_logger()
        hf_logging._configure_library_root_logger()
        hf_root = hf_logging._get_library_root_logger()
        try:
            assert hf_root.propagate is False
            assert any(isinstance(h, logging.StreamHandler) for h in hf_root.handlers)

            install_transformers_logging_bridge()

            assert hf_root.propagate is True
            assert hf_root.handlers == []
        finally:
            hf_logging._reset_library_root_logger()
            hf_logging._configure_library_root_logger()

    def test_bridge_lets_vocab_token_warning_be_filtered(self):
        import transformers.utils.logging as hf_logging

        stream = io.StringIO()
        try:
            setup_logging(level="DEBUG", fmt="text", stream=stream)
            install_transformers_logging_bridge()
            logging.getLogger("transformers.configuration_utils").warning(
                "Model config: bos_token_id must be `None` or an integer within "
                "the vocabulary (between 0 and 31999), got 49406."
            )
            assert stream.getvalue() == ""
        finally:
            hf_logging._reset_library_root_logger()
            hf_logging._configure_library_root_logger()
            setup_logging()


class TestTransformersImportIsNotAStatWalk:
    """The bridge seeds the stat-free ``packages_distributions`` first.

    transformers builds the mapping at module import, and the stdlib version of
    it stats every file recorded by every installed distribution - 16 minutes of
    silent startup on a cold NFS venv (issue #3715).
    """

    def test_bridge_seeds_before_importing_transformers(self):
        import importlib.metadata

        from vtscore.utils.import_metadata import fast_packages_distributions

        stdlib = importlib.metadata.packages_distributions
        called = []

        def tripwire():
            called.append(True)
            return {}

        importlib.metadata.packages_distributions = tripwire
        try:
            install_transformers_logging_bridge()
            assert called == [], "startup walked every recorded file of every distribution"
            assert importlib.metadata.packages_distributions is fast_packages_distributions
        finally:
            importlib.metadata.packages_distributions = stdlib
            hf_logging = importlib.import_module("transformers.utils.logging")
            hf_logging._reset_library_root_logger()
            hf_logging._configure_library_root_logger()


class TestRequestIdHelper:
    def test_new_request_id_is_unique(self):
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_new_request_id_length(self):
        rid = new_request_id()
        assert len(rid) == 12
        # Hex characters only.
        int(rid, 16)


# ---------------------------------------------------------------------------
# ContextFilter does not overwrite caller-supplied extras
# ---------------------------------------------------------------------------


class TestExtraOverride:
    def test_caller_supplied_extras_win(self):
        """If a caller passes extra={"dataset_id": "X"}, don't clobber it
        with the resolved context (which may be None outside a request)."""
        stream = io.StringIO()

        def go():
            logging.getLogger("vtsearch.test").info("explicit", extra={"dataset_id": "explicit-ds"})

        records = _emit(stream, "json", go)
        assert records[0]["dataset_id"] == "explicit-ds"
