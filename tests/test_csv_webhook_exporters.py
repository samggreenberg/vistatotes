"""Tests for CSV and Webhook exporters."""

import argparse
import csv
import json
from unittest import mock

import pytest

import app as app_module
from vistatotes.cli import (
    _build_results_dict,
    _run_exporter,
    run_autodetect,
)
from vistatotes.datasets.loader import export_dataset_to_file


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_cli_autodetect.py)
# ---------------------------------------------------------------------------


def _make_dataset_file(tmp_path, clips_dict):
    """Export a clips dict to a pickle file and return the path."""
    pkl_bytes = export_dataset_to_file(clips_dict)
    dataset_path = tmp_path / "dataset.pkl"
    dataset_path.write_bytes(pkl_bytes)
    return dataset_path


def _make_detector_file(tmp_path, client, good_ids, bad_ids, name="detector.json"):
    """Train a detector via the API and write its JSON to a file."""
    app_module.good_votes.update({k: None for k in good_ids})
    app_module.bad_votes.update({k: None for k in bad_ids})
    resp = client.post("/api/detector/export")
    assert resp.status_code == 200
    detector = resp.get_json()
    app_module.good_votes.clear()
    app_module.bad_votes.clear()

    detector_path = tmp_path / name
    detector_path.write_text(json.dumps(detector))
    return detector_path, detector


def _make_sample_results():
    """Create a sample results dict for testing exporters directly."""
    return {
        "media_type": "audio",
        "detectors_run": 1,
        "results": {
            "test_detector": {
                "detector_name": "test_detector",
                "threshold": 0.5,
                "total_hits": 3,
                "hits": [
                    {"id": 1, "filename": "clip_1.wav", "category": "birds", "score": 0.95},
                    {"id": 2, "filename": "clip_2.wav", "category": "rain", "score": 0.82},
                    {"id": 3, "filename": "clip_3.wav", "category": "birds", "score": 0.71},
                ],
            }
        },
    }


# ---------------------------------------------------------------------------
# CSV Exporter — metadata
# ---------------------------------------------------------------------------


class TestCsvExporterMetadata:
    """CSV exporter metadata and registration."""

    def test_csv_exporter_registered(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("csv")
        assert exp is not None

    def test_csv_exporter_display_name(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("csv")
        assert exp.display_name == "Save to CSV"

    def test_csv_exporter_icon(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("csv")
        assert exp.icon == "\U0001f4ca"

    def test_csv_exporter_has_filepath_field(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("csv")
        keys = [f.key for f in exp.fields]
        assert "filepath" in keys

    def test_csv_exporter_to_dict(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("csv")
        d = exp.to_dict()
        assert d["name"] == "csv"
        assert "fields" in d
        assert len(d["fields"]) >= 1


# ---------------------------------------------------------------------------
# CSV Exporter — CLI arguments
# ---------------------------------------------------------------------------


class TestCsvExporterCLI:
    """CLI argument parsing for the CSV exporter."""

    def test_adds_filepath_arg(self):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args(["--filepath", "/tmp/results.csv"])
        assert args.filepath == "/tmp/results.csv"

    def test_filepath_default(self):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args([])
        assert args.filepath == "autodetect_results.csv"

    def test_validate_passes(self):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        exp.validate_cli_field_values({"filepath": "/tmp/out.csv"})

    def test_validate_missing_filepath(self):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        with pytest.raises(ValueError, match="Missing required argument: --filepath"):
            exp.validate_cli_field_values({})


# ---------------------------------------------------------------------------
# CSV Exporter — export functionality
# ---------------------------------------------------------------------------


class TestCsvExporterExport:
    """Tests for the CSV export() method."""

    def test_creates_csv_file(self, tmp_path):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        results = _make_sample_results()
        filepath = tmp_path / "output.csv"

        result = exp.export(results, {"filepath": str(filepath)})
        assert filepath.exists()
        assert "message" in result
        assert "Saved" in result["message"]

    def test_csv_has_correct_header(self, tmp_path):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        results = _make_sample_results()
        filepath = tmp_path / "output.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["detector", "threshold", "filename", "category", "score"]

    def test_csv_has_correct_row_count(self, tmp_path):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        results = _make_sample_results()
        filepath = tmp_path / "output.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 1 header + 3 data rows
        assert len(rows) == 4

    def test_csv_row_values(self, tmp_path):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        results = _make_sample_results()
        filepath = tmp_path / "output.csv"

        exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            first_row = next(reader)
        assert first_row[0] == "test_detector"
        assert first_row[2] == "clip_1.wav"
        assert first_row[3] == "birds"
        assert first_row[4] == "0.95"

    def test_csv_empty_results(self, tmp_path):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        results = {"media_type": "audio", "detectors_run": 0, "results": {}}
        filepath = tmp_path / "empty.csv"

        result = exp.export(results, {"filepath": str(filepath)})
        assert filepath.exists()
        assert "0 hit(s)" in result["message"]

    def test_csv_creates_parent_dirs(self, tmp_path):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        results = _make_sample_results()
        filepath = tmp_path / "sub" / "dir" / "output.csv"

        exp.export(results, {"filepath": str(filepath)})
        assert filepath.exists()

    def test_csv_empty_filepath_raises(self):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        with pytest.raises(ValueError, match="file path is required"):
            exp.export({}, {"filepath": ""})

    def test_csv_multiple_detectors(self, tmp_path):
        from vistatotes.exporters.csv_file import CsvExporter

        exp = CsvExporter()
        results = {
            "media_type": "audio",
            "detectors_run": 2,
            "results": {
                "det_a": {
                    "detector_name": "det_a",
                    "threshold": 0.4,
                    "total_hits": 1,
                    "hits": [{"id": 1, "filename": "a.wav", "category": "x", "score": 0.9}],
                },
                "det_b": {
                    "detector_name": "det_b",
                    "threshold": 0.6,
                    "total_hits": 2,
                    "hits": [
                        {"id": 2, "filename": "b.wav", "category": "y", "score": 0.8},
                        {"id": 3, "filename": "c.wav", "category": "z", "score": 0.7},
                    ],
                },
            },
        }
        filepath = tmp_path / "multi.csv"

        result = exp.export(results, {"filepath": str(filepath)})
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 1 header + 3 data rows
        assert len(rows) == 4
        assert "3 hit(s)" in result["message"]
        assert "2 detector(s)" in result["message"]


class TestCsvExporterIntegration:
    """Integration: CSV exporter via _run_exporter and with real detector results."""

    def test_csv_via_run_exporter(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        results = _build_results_dict(hits, str(detector_path), "audio")

        output_file = tmp_path / "integrated.csv"
        _run_exporter("csv", {"filepath": str(output_file)}, results)

        assert output_file.exists()
        with open(output_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header[0] == "detector"


# ---------------------------------------------------------------------------
# Webhook Exporter — metadata
# ---------------------------------------------------------------------------


class TestWebhookExporterMetadata:
    """Webhook exporter metadata and registration."""

    def test_webhook_exporter_registered(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("webhook")
        assert exp is not None

    def test_webhook_exporter_display_name(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("webhook")
        assert exp.display_name == "Webhook (HTTP POST)"

    def test_webhook_exporter_icon(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("webhook")
        assert exp.icon == "\U0001f310"

    def test_webhook_exporter_has_url_field(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("webhook")
        keys = [f.key for f in exp.fields]
        assert "url" in keys

    def test_webhook_exporter_has_auth_header_field(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("webhook")
        keys = [f.key for f in exp.fields]
        assert "auth_header" in keys

    def test_webhook_auth_header_not_required(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("webhook")
        auth_field = next(f for f in exp.fields if f.key == "auth_header")
        assert auth_field.required is False

    def test_webhook_exporter_to_dict(self):
        from vistatotes.exporters import get_exporter

        exp = get_exporter("webhook")
        d = exp.to_dict()
        assert d["name"] == "webhook"
        assert "fields" in d


# ---------------------------------------------------------------------------
# Webhook Exporter — CLI arguments
# ---------------------------------------------------------------------------


class TestWebhookExporterCLI:
    """CLI argument parsing for the Webhook exporter."""

    def test_adds_url_arg(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args(["--url", "https://example.com/hook"])
        assert args.url == "https://example.com/hook"

    def test_validate_passes_with_url(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        # auth_header is optional, so only url is needed
        exp.validate_cli_field_values({"url": "https://example.com/hook"})

    def test_validate_missing_url(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        with pytest.raises(ValueError, match="Missing required argument: --url"):
            exp.validate_cli_field_values({})

    def test_validate_passes_without_auth_header(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        # Should not raise
        exp.validate_cli_field_values({"url": "https://example.com/hook"})


# ---------------------------------------------------------------------------
# Webhook Exporter — export functionality
# ---------------------------------------------------------------------------


class TestWebhookExporterExport:
    """Tests for the Webhook export() method using mocked HTTP."""

    def test_posts_json_to_url(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with mock.patch("vistatotes.exporters.webhook.requests.post", return_value=mock_resp) as mock_post:
            result = exp.export(results, {"url": "https://example.com/hook"})

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"] == results
        assert "message" in result
        assert "200" in result["message"]

    def test_sends_auth_header_when_provided(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with mock.patch("vistatotes.exporters.webhook.requests.post", return_value=mock_resp) as mock_post:
            exp.export(results, {"url": "https://example.com/hook", "auth_header": "Bearer my-token"})

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-token"

    def test_no_auth_header_when_empty(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with mock.patch("vistatotes.exporters.webhook.requests.post", return_value=mock_resp) as mock_post:
            exp.export(results, {"url": "https://example.com/hook", "auth_header": ""})

        call_kwargs = mock_post.call_args
        assert "Authorization" not in call_kwargs.kwargs["headers"]

    def test_empty_url_raises(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        with pytest.raises(ValueError, match="webhook URL is required"):
            exp.export({}, {"url": ""})

    def test_http_error_propagates(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")

        with mock.patch("vistatotes.exporters.webhook.requests.post", return_value=mock_resp):
            with pytest.raises(Exception, match="500 Server Error"):
                exp.export(results, {"url": "https://example.com/hook"})

    def test_message_contains_hit_count(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with mock.patch("vistatotes.exporters.webhook.requests.post", return_value=mock_resp):
            result = exp.export(results, {"url": "https://example.com/hook"})

        assert "3 hit(s)" in result["message"]
        assert "1 detector(s)" in result["message"]

    def test_result_includes_status_code_and_url(self):
        from vistatotes.exporters.webhook import WebhookExporter

        exp = WebhookExporter()
        results = _make_sample_results()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 201
        mock_resp.raise_for_status.return_value = None

        with mock.patch("vistatotes.exporters.webhook.requests.post", return_value=mock_resp):
            result = exp.export(results, {"url": "https://example.com/hook"})

        assert result["status_code"] == 201
        assert result["url"] == "https://example.com/hook"


class TestWebhookExporterIntegration:
    """Integration: Webhook exporter via _run_exporter."""

    def test_webhook_via_run_exporter(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        results = _build_results_dict(hits, str(detector_path), "audio")

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None

        with mock.patch("vistatotes.exporters.webhook.requests.post", return_value=mock_resp):
            _run_exporter("webhook", {"url": "https://example.com/hook"}, results)

    def test_unknown_exporter_still_raises(self):
        with pytest.raises(ValueError, match="Unknown exporter"):
            _run_exporter("nonexistent_exporter", {}, {})
