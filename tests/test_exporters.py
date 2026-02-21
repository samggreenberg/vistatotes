"""Tests for the Labelset Exporter abstraction.

Covers:
- ExporterField and LabelsetExporter base classes
- Auto-discovery registry
- Built-in exporters: gui, file, email_smtp
- Flask API routes: GET /api/exporters, POST /api/exporters/export
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SAMPLE_RESULTS = {
    "media_type": "audio",
    "detectors_run": 2,
    "results": {
        "dog_bark": {
            "detector_name": "dog_bark",
            "threshold": 0.5,
            "total_hits": 3,
            "hits": [
                {"id": 1, "filename": "bark1.wav", "score": 0.9},
                {"id": 2, "filename": "bark2.wav", "score": 0.7},
                {"id": 3, "filename": "bark3.wav", "score": 0.6},
            ],
        },
        "cat_meow": {
            "detector_name": "cat_meow",
            "threshold": 0.6,
            "total_hits": 1,
            "hits": [
                {"id": 5, "filename": "meow.wav", "score": 0.8},
            ],
        },
    },
}

EMPTY_RESULTS = {
    "media_type": "audio",
    "detectors_run": 0,
    "results": {},
}


# ---------------------------------------------------------------------------
# ExporterField
# ---------------------------------------------------------------------------


class TestExporterField:
    def test_to_dict_contains_required_keys(self):
        from vtsearch.exporters.base import ExporterField

        f = ExporterField(key="fp", label="File Path", field_type="text")
        d = f.to_dict()
        assert d["key"] == "fp"
        assert d["label"] == "File Path"
        assert d["field_type"] == "text"
        assert "description" in d
        assert "options" in d
        assert "default" in d
        assert "required" in d
        assert "placeholder" in d

    def test_defaults(self):
        from vtsearch.exporters.base import ExporterField

        f = ExporterField(key="x", label="X", field_type="text")
        assert f.required is True
        assert f.default == ""
        assert f.placeholder == ""
        assert f.options == []
        assert f.description == ""

    def test_custom_values(self):
        from vtsearch.exporters.base import ExporterField

        f = ExporterField(
            key="mode",
            label="Mode",
            field_type="select",
            options=["a", "b"],
            default="a",
            required=False,
            description="Choose mode",
            placeholder="Pick one",
        )
        d = f.to_dict()
        assert d["options"] == ["a", "b"]
        assert d["default"] == "a"
        assert d["required"] is False


# ---------------------------------------------------------------------------
# LabelsetExporter base class
# ---------------------------------------------------------------------------


class TestLabelsetExporterBase:
    def test_export_raises_not_implemented(self):
        from vtsearch.exporters.base import LabelsetExporter

        exp = LabelsetExporter()
        with pytest.raises(NotImplementedError):
            exp.export({}, {})

    def test_to_dict_contains_standard_keys(self):
        from vtsearch.exporters.base import ExporterField, LabelsetExporter

        class Dummy(LabelsetExporter):
            name = "dummy"
            display_name = "Dummy"
            description = "A test exporter."
            icon = "🧪"
            fields = [ExporterField(key="k", label="K", field_type="text")]

            def export(self, results, field_values):
                return {"message": "ok"}

        d = Dummy().to_dict()
        assert d["name"] == "dummy"
        assert d["display_name"] == "Dummy"
        assert d["description"] == "A test exporter."
        assert d["icon"] == "🧪"
        assert len(d["fields"]) == 1
        assert d["fields"][0]["key"] == "k"


# ---------------------------------------------------------------------------
# Registry (auto-discovery)
# ---------------------------------------------------------------------------


class TestExporterRegistry:
    def test_list_exporters_returns_all_builtins(self):
        from vtsearch.exporters import list_exporters

        names = {e.name for e in list_exporters()}
        assert "gui" in names
        assert "file" in names
        assert "email_smtp" in names

    def test_get_exporter_known(self):
        from vtsearch.exporters import get_exporter

        for name in ("gui", "file", "email_smtp"):
            exp = get_exporter(name)
            assert exp is not None, f"Exporter '{name}' not found"
            assert exp.name == name

    def test_get_exporter_unknown_returns_none(self):
        from vtsearch.exporters import get_exporter

        assert get_exporter("no_such_exporter") is None

    def test_each_exporter_has_display_name_and_icon(self):
        from vtsearch.exporters import list_exporters

        for exp in list_exporters():
            assert exp.display_name, f"{exp.name} missing display_name"
            assert exp.icon, f"{exp.name} missing icon"
            assert exp.description, f"{exp.name} missing description"

    def test_each_exporter_fields_are_valid(self):
        from vtsearch.exporters import list_exporters

        for exp in list_exporters():
            for f in exp.fields:
                assert f.key, f"{exp.name} has a field without a key"
                assert f.label, f"{exp.name} field '{f.key}' has no label"
                assert f.field_type in ("text", "password", "email", "file", "folder", "select"), (
                    f"{exp.name} field '{f.key}' has unknown type '{f.field_type}'"
                )


# ---------------------------------------------------------------------------
# GUI exporter
# ---------------------------------------------------------------------------


class TestDisplayLabelsetExporter:
    def test_has_no_fields(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("gui")
        assert exp.fields == []

    def test_export_returns_message_and_display_results(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("gui")
        result = exp.export(SAMPLE_RESULTS, {})
        assert "message" in result
        assert "display_results" in result
        assert result["display_results"] is SAMPLE_RESULTS

    def test_export_counts_hits_in_message(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("gui")
        result = exp.export(SAMPLE_RESULTS, {})
        # 3 + 1 = 4 total hits
        assert "4" in result["message"]
        assert "2" in result["message"]  # 2 detectors

    def test_export_empty_results(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("gui")
        result = exp.export(EMPTY_RESULTS, {})
        assert "message" in result
        assert result["display_results"] is EMPTY_RESULTS

    def test_export_cli_prints_origins_and_names(self, capsys):
        from vtsearch.exporters import get_exporter

        results_with_origin = {
            "media_type": "audio",
            "detectors_run": 1,
            "results": {
                "det1": {
                    "detector_name": "det1",
                    "threshold": 0.5,
                    "total_hits": 2,
                    "hits": [
                        {
                            "id": 1,
                            "filename": "bark1.wav",
                            "origin_name": "bark1.wav",
                            "origin": {"importer": "folder", "params": {"path": "/data"}},
                            "score": 0.9,
                            "category": "dog",
                        },
                        {
                            "id": 2,
                            "filename": "bark2.wav",
                            "origin_name": "bark2.wav",
                            "score": 0.7,
                            "category": "dog",
                        },
                    ],
                },
            },
        }
        exp = get_exporter("gui")
        result = exp.export_cli(results_with_origin, {})
        captured = capsys.readouterr()
        assert "message" in result
        # Should list origin and name, not scores or categories
        assert "folder(/data)" in captured.out
        assert "bark1.wav" in captured.out
        assert "bark2.wav" in captured.out
        assert "score" not in captured.out.lower()
        assert "category" not in captured.out.lower()
        assert "dog" not in captured.out

    def test_export_cli_no_hits(self, capsys):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("gui")
        result = exp.export_cli(EMPTY_RESULTS, {})
        captured = capsys.readouterr()
        assert "No items predicted as Good" in captured.out
        assert "message" in result


# ---------------------------------------------------------------------------
# File exporter
# ---------------------------------------------------------------------------


class TestFileLabelsetExporter:
    def test_has_filepath_field(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("file")
        keys = [f.key for f in exp.fields]
        assert "filepath" in keys

    def test_export_writes_json(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("file")
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "results.json"
            result = exp.export(SAMPLE_RESULTS, {"filepath": str(fpath)})
            assert "message" in result
            assert fpath.exists()
            written = json.loads(fpath.read_text())
            assert written["media_type"] == "audio"
            assert written["detectors_run"] == 2

    def test_export_creates_parent_dirs(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("file")
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "sub" / "dir" / "results.json"
            exp.export(SAMPLE_RESULTS, {"filepath": str(fpath)})
            assert fpath.exists()

    def test_export_raises_on_empty_filepath(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("file")
        with pytest.raises(ValueError, match="file path"):
            exp.export(SAMPLE_RESULTS, {"filepath": ""})

    def test_export_message_contains_hit_count(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("file")
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "out.json"
            result = exp.export(SAMPLE_RESULTS, {"filepath": str(fpath)})
            assert "4" in result["message"]  # 3 + 1 hits

    def test_to_dict_has_all_keys(self):
        from vtsearch.exporters import get_exporter

        d = get_exporter("file").to_dict()
        assert d["name"] == "file"
        assert "fields" in d
        assert len(d["fields"]) >= 1


# ---------------------------------------------------------------------------
# Email SMTP exporter
# ---------------------------------------------------------------------------


class TestEmailLabelsetExporter:
    def test_has_required_fields(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("email_smtp")
        keys = {f.key for f in exp.fields}
        assert "to" in keys
        assert "from_email" in keys
        assert "smtp_password" in keys
        assert "smtp_host" in keys
        assert "smtp_port" in keys

    def test_password_field_type_is_password(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("email_smtp")
        pwd_field = next(f for f in exp.fields if f.key == "smtp_password")
        assert pwd_field.field_type == "password"

    def test_export_raises_on_missing_to(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("email_smtp")
        with pytest.raises(ValueError, match="Recipient"):
            exp.export(
                SAMPLE_RESULTS,
                {
                    "to": "",
                    "from_email": "me@example.com",
                    "smtp_password": "secret",
                    "smtp_host": "smtp.example.com",
                    "smtp_port": "587",
                },
            )

    def test_export_raises_on_missing_from(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("email_smtp")
        with pytest.raises(ValueError, match="Sender"):
            exp.export(
                SAMPLE_RESULTS,
                {
                    "to": "you@example.com",
                    "from_email": "",
                    "smtp_password": "secret",
                    "smtp_host": "smtp.example.com",
                    "smtp_port": "587",
                },
            )

    def test_export_raises_on_missing_password(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("email_smtp")
        with pytest.raises(ValueError, match="password"):
            exp.export(
                SAMPLE_RESULTS,
                {
                    "to": "you@example.com",
                    "from_email": "me@example.com",
                    "smtp_password": "",
                    "smtp_host": "smtp.example.com",
                    "smtp_port": "587",
                },
            )

    def test_export_calls_smtp(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("email_smtp")

        mock_server = MagicMock()
        mock_smtp_cls = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with patch("vtsearch.exporters.email_smtp.smtplib.SMTP", mock_smtp_cls):
            result = exp.export(
                SAMPLE_RESULTS,
                {
                    "to": "you@example.com",
                    "from_email": "me@example.com",
                    "smtp_password": "secret",
                    "smtp_host": "smtp.example.com",
                    "smtp_port": "587",
                },
            )

        mock_smtp_cls.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("me@example.com", "secret")
        mock_server.sendmail.assert_called_once()
        assert "message" in result
        assert "you@example.com" in result["message"]

    def test_plain_text_builder(self):
        from vtsearch.exporters.email_smtp import _build_plain_text

        text = _build_plain_text(SAMPLE_RESULTS)
        assert "Auto-Detect Results" in text
        assert "dog_bark" in text
        assert "cat_meow" in text
        assert "bark1.wav" in text

    def test_html_builder(self):
        from vtsearch.exporters.email_smtp import _build_html

        html = _build_html(SAMPLE_RESULTS)
        assert "<html>" in html
        assert "dog_bark" in html
        assert "cat_meow" in html
        assert "bark1.wav" in html

    def test_default_smtp_host_in_field(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("email_smtp")
        host_field = next(f for f in exp.fields if f.key == "smtp_host")
        assert host_field.default == "smtp.gmail.com"

    def test_default_smtp_port_in_field(self):
        from vtsearch.exporters import get_exporter

        exp = get_exporter("email_smtp")
        port_field = next(f for f in exp.fields if f.key == "smtp_port")
        assert port_field.default == "587"


# ---------------------------------------------------------------------------
# API – GET /api/exporters
# ---------------------------------------------------------------------------


class TestGetExportersEndpoint:
    def test_returns_200(self, client):
        res = client.get("/api/exporters")
        assert res.status_code == 200

    def test_returns_list(self, client):
        res = client.get("/api/exporters")
        data = res.get_json()
        assert isinstance(data, list)

    def test_contains_builtin_exporters(self, client):
        res = client.get("/api/exporters")
        names = {e["name"] for e in res.get_json()}
        assert "gui" in names
        assert "file" in names
        assert "email_smtp" in names

    def test_each_entry_has_required_keys(self, client):
        res = client.get("/api/exporters")
        for entry in res.get_json():
            assert "name" in entry
            assert "display_name" in entry
            assert "description" in entry
            assert "icon" in entry
            assert "fields" in entry


# ---------------------------------------------------------------------------
# API – POST /api/exporters/export
# ---------------------------------------------------------------------------


class TestExportEndpoint:
    def test_missing_exporter_name_returns_400(self, client):
        res = client.post(
            "/api/exporters/export",
            json={"results": SAMPLE_RESULTS},
        )
        assert res.status_code == 400
        assert "exporter_name" in res.get_json()["error"]

    def test_unknown_exporter_returns_404(self, client):
        res = client.post(
            "/api/exporters/export",
            json={"exporter_name": "unicorn", "results": SAMPLE_RESULTS},
        )
        assert res.status_code == 404
        assert "unicorn" in res.get_json()["error"]

    def test_gui_exporter_returns_success(self, client):
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "gui",
                "field_values": {},
                "results": SAMPLE_RESULTS,
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "message" in data
        assert "display_results" in data

    def test_file_exporter_creates_file(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "export.json"
            res = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "file",
                    "field_values": {"filepath": str(fpath)},
                    "results": SAMPLE_RESULTS,
                },
            )
            assert res.status_code == 200
            data = res.get_json()
            assert data["success"] is True
            assert fpath.exists()
            written = json.loads(fpath.read_text())
            assert written["detectors_run"] == 2

    def test_file_exporter_missing_filepath_returns_400(self, client):
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "file",
                "field_values": {"filepath": ""},
                "results": SAMPLE_RESULTS,
            },
        )
        assert res.status_code == 400

    def test_file_exporter_missing_field_returns_400(self, client):
        """The route should reject the request before calling export()."""
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "file",
                "field_values": {},  # 'filepath' is required but absent
                "results": SAMPLE_RESULTS,
            },
        )
        assert res.status_code == 400
        data = res.get_json()
        assert "missing_fields" in data
        assert "filepath" in data["missing_fields"]

    def test_email_exporter_sends_via_smtp(self, client):
        mock_server = MagicMock()
        mock_smtp_cls = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with patch("vtsearch.exporters.email_smtp.smtplib.SMTP", mock_smtp_cls):
            res = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "email_smtp",
                    "field_values": {
                        "to": "you@example.com",
                        "from_email": "me@example.com",
                        "smtp_password": "secret",
                        "smtp_host": "smtp.example.com",
                        "smtp_port": "587",
                    },
                    "results": SAMPLE_RESULTS,
                },
            )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "you@example.com" in data["message"]
        mock_server.sendmail.assert_called_once()

    def test_export_with_empty_results_dict(self, client):
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "gui",
                "field_values": {},
                "results": {},
            },
        )
        assert res.status_code == 200
        assert res.get_json()["success"] is True

    def test_export_with_no_results_key(self, client):
        """results defaults to {} when omitted."""
        res = client.post(
            "/api/exporters/export",
            json={"exporter_name": "gui"},
        )
        assert res.status_code == 200

    def test_non_json_body_treated_as_empty(self, client):
        res = client.post(
            "/api/exporters/export",
            data="not json",
            content_type="text/plain",
        )
        # exporter_name will be empty → 400
        assert res.status_code == 400
