"""Tests for the Label Importer abstraction.

Covers:
- LabelImporterField and LabelImporter base classes
- Auto-discovery registry (list_label_importers, get_label_importer)
- Built-in importers: json_file, csv_file
- Flask API routes: GET /api/label-importers, POST /api/label-importers/import/<name>
- CLI import_labels_main function
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import app as app_module


# ---------------------------------------------------------------------------
# LabelImporterField
# ---------------------------------------------------------------------------


class TestLabelImporterField:
    def test_to_dict_contains_required_keys(self):
        from vtsearch.labels.importers.base import LabelImporterField

        f = LabelImporterField(key="file", label="My File", field_type="file")
        d = f.to_dict()
        assert d["key"] == "file"
        assert d["label"] == "My File"
        assert d["field_type"] == "file"
        assert "description" in d
        assert "accept" in d
        assert "options" in d
        assert "default" in d
        assert "required" in d
        assert "placeholder" in d

    def test_defaults(self):
        from vtsearch.labels.importers.base import LabelImporterField

        f = LabelImporterField(key="x", label="X", field_type="text")
        assert f.required is True
        assert f.default == ""
        assert f.placeholder == ""
        assert f.options == []
        assert f.description == ""
        assert f.accept == ""

    def test_custom_values(self):
        from vtsearch.labels.importers.base import LabelImporterField

        f = LabelImporterField(
            key="mode",
            label="Mode",
            field_type="select",
            options=["a", "b"],
            default="a",
            required=False,
            description="Pick one",
            placeholder="Choose…",
        )
        d = f.to_dict()
        assert d["options"] == ["a", "b"]
        assert d["default"] == "a"
        assert d["required"] is False


# ---------------------------------------------------------------------------
# LabelImporter base class
# ---------------------------------------------------------------------------


class TestLabelImporterBase:
    def _make_minimal(self):
        from vtsearch.labels.importers.base import LabelImporter

        class Minimal(LabelImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "A minimal label importer."
            fields = []

            def run(self, field_values):
                return []

        return Minimal()

    def test_run_raises_not_implemented_when_not_overridden(self):
        from vtsearch.labels.importers.base import LabelImporter

        imp = LabelImporter()
        with pytest.raises(NotImplementedError):
            imp.run({})

    def test_to_dict_contains_standard_keys(self):
        imp = self._make_minimal()
        d = imp.to_dict()
        assert d["name"] == "minimal"
        assert d["display_name"] == "Minimal"
        assert d["description"] == "A minimal label importer."
        assert "icon" in d
        assert "fields" in d

    def test_default_icon(self):
        from vtsearch.labels.importers.base import LabelImporter

        assert LabelImporter.icon == "🏷️"

    def test_custom_icon_in_to_dict(self):
        from vtsearch.labels.importers.base import LabelImporter

        class Custom(LabelImporter):
            name = "c"
            display_name = "C"
            description = "C"
            icon = "🔖"
            fields = []

            def run(self, field_values):
                return []

        assert Custom().to_dict()["icon"] == "🔖"

    def test_validate_cli_field_values_raises_on_missing_required(self):
        from vtsearch.labels.importers.base import LabelImporter, LabelImporterField

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [LabelImporterField("filepath", "File", "text", required=True)]

            def run(self, field_values):
                return []

        imp = Imp()
        with pytest.raises(ValueError, match="--filepath"):
            imp.validate_cli_field_values({})

    def test_validate_cli_field_values_passes_when_provided(self):
        from vtsearch.labels.importers.base import LabelImporter, LabelImporterField

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [LabelImporterField("filepath", "File", "text", required=True)]

            def run(self, field_values):
                return []

        imp = Imp()
        imp.validate_cli_field_values({"filepath": "/some/path"})  # no raise

    def test_run_cli_delegates_to_run(self):
        from vtsearch.labels.importers.base import LabelImporter

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values):
                return [{"md5": "abc", "label": "good"}]

        imp = Imp()
        result = imp.run_cli({})
        assert result == [{"md5": "abc", "label": "good"}]

    def test_add_cli_arguments_adds_text_field(self):
        import argparse

        from vtsearch.labels.importers.base import LabelImporter, LabelImporterField

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [LabelImporterField("server", "Server", "text", description="DB host")]

            def run(self, field_values):
                return []

        parser = argparse.ArgumentParser()
        Imp().add_cli_arguments(parser)
        args = parser.parse_args(["--server", "localhost"])
        assert args.server == "localhost"

    def test_add_cli_arguments_select_adds_choices(self):
        import argparse

        from vtsearch.labels.importers.base import LabelImporter, LabelImporterField

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [LabelImporterField("mode", "Mode", "select", options=["a", "b"], default="a")]

            def run(self, field_values):
                return []

        parser = argparse.ArgumentParser()
        Imp().add_cli_arguments(parser)
        args = parser.parse_args([])  # uses default
        assert args.mode == "a"
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestLabelImporterRegistry:
    def test_list_label_importers_returns_builtins(self):
        from vtsearch.labels.importers import list_label_importers

        names = {imp.name for imp in list_label_importers()}
        assert "json_file" in names
        assert "csv_file" in names

    def test_get_label_importer_known(self):
        from vtsearch.labels.importers import get_label_importer

        for name in ("json_file", "csv_file"):
            imp = get_label_importer(name)
            assert imp is not None, f"Label importer '{name}' not found"
            assert imp.name == name

    def test_get_label_importer_unknown_returns_none(self):
        from vtsearch.labels.importers import get_label_importer

        assert get_label_importer("no_such_importer") is None

    def test_each_importer_has_display_name_and_icon(self):
        from vtsearch.labels.importers import list_label_importers

        for imp in list_label_importers():
            assert imp.display_name, f"{imp.name} missing display_name"
            assert imp.icon, f"{imp.name} missing icon"
            assert imp.description, f"{imp.name} missing description"

    def test_each_importer_fields_are_valid(self):
        from vtsearch.labels.importers import list_label_importers

        valid_types = ("file", "text", "password", "select")
        for imp in list_label_importers():
            for f in imp.fields:
                assert f.key, f"{imp.name} has a field without a key"
                assert f.label, f"{imp.name} field '{f.key}' has no label"
                assert f.field_type in valid_types, (
                    f"{imp.name} field '{f.key}' has unknown type '{f.field_type}'"
                )


# ---------------------------------------------------------------------------
# JSON file importer
# ---------------------------------------------------------------------------


class TestJsonLabelImporter:
    def _get_importer(self):
        from vtsearch.labels.importers.json_file import LABEL_IMPORTER

        return LABEL_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "json_file"

    def test_display_name(self):
        assert "json" in self._get_importer().display_name.lower()

    def test_icon(self):
        assert self._get_importer().icon

    def test_has_file_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "file" in fields
        assert fields["file"].field_type == "file"

    def test_run_with_file_storage(self):
        from werkzeug.datastructures import FileStorage

        payload = {"labels": [{"md5": "abc", "label": "good"}]}
        raw = json.dumps(payload).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="labels.json", content_type="application/json")
        result = self._get_importer().run({"file": fs})
        assert len(result) == 1
        assert result[0]["md5"] == "abc"
        assert result[0]["label"] == "good"

    def test_run_returns_both_good_and_bad(self):
        from werkzeug.datastructures import FileStorage

        payload = {
            "labels": [
                {"md5": "aaa", "label": "good"},
                {"md5": "bbb", "label": "bad"},
            ]
        }
        raw = json.dumps(payload).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="labels.json")
        result = self._get_importer().run({"file": fs})
        labels = {r["md5"]: r["label"] for r in result}
        assert labels["aaa"] == "good"
        assert labels["bbb"] == "bad"

    def test_run_raises_on_invalid_json(self):
        from werkzeug.datastructures import FileStorage

        fs = FileStorage(stream=io.BytesIO(b"not json at all"), filename="labels.json")
        with pytest.raises(ValueError, match="JSON"):
            self._get_importer().run({"file": fs})

    def test_run_raises_on_missing_labels_key(self):
        from werkzeug.datastructures import FileStorage

        fs = FileStorage(stream=io.BytesIO(b'{"data": []}'), filename="labels.json")
        with pytest.raises(ValueError, match="labels"):
            self._get_importer().run({"file": fs})

    def test_run_raises_when_no_file(self):
        with pytest.raises(ValueError):
            self._get_importer().run({"file": None})

    def test_run_cli_reads_file_path(self, tmp_path):
        payload = {"labels": [{"md5": "xyz", "label": "bad"}]}
        p = tmp_path / "labels.json"
        p.write_text(json.dumps(payload))
        result = self._get_importer().run_cli({"file": str(p)})
        assert len(result) == 1
        assert result[0]["md5"] == "xyz"

    def test_run_cli_raises_on_empty_path(self):
        with pytest.raises(ValueError, match="--file"):
            self._get_importer().run_cli({"file": ""})


# ---------------------------------------------------------------------------
# CSV file importer
# ---------------------------------------------------------------------------


class TestCsvLabelImporter:
    def _get_importer(self):
        from vtsearch.labels.importers.csv_file import LABEL_IMPORTER

        return LABEL_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "csv_file"

    def test_display_name(self):
        assert "csv" in self._get_importer().display_name.lower()

    def test_icon(self):
        assert self._get_importer().icon

    def test_has_file_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "file" in fields
        assert fields["file"].field_type == "file"

    def test_run_with_file_storage(self):
        from werkzeug.datastructures import FileStorage

        csv_bytes = b"md5,label\nabc123,good\ndef456,bad\n"
        fs = FileStorage(stream=io.BytesIO(csv_bytes), filename="labels.csv", content_type="text/csv")
        result = self._get_importer().run({"file": fs})
        assert len(result) == 2
        labels = {r["md5"]: r["label"] for r in result}
        assert labels["abc123"] == "good"
        assert labels["def456"] == "bad"

    def test_run_strips_bom(self):
        from werkzeug.datastructures import FileStorage

        csv_bytes = b"\xef\xbb\xbfmd5,label\nabc,good\n"
        fs = FileStorage(stream=io.BytesIO(csv_bytes), filename="labels.csv")
        result = self._get_importer().run({"file": fs})
        assert len(result) == 1
        assert result[0]["md5"] == "abc"

    def test_run_normalises_label_to_lowercase(self):
        from werkzeug.datastructures import FileStorage

        csv_bytes = b"md5,label\nabc,GOOD\n"
        fs = FileStorage(stream=io.BytesIO(csv_bytes), filename="labels.csv")
        result = self._get_importer().run({"file": fs})
        assert result[0]["label"] == "good"

    def test_run_raises_on_missing_columns(self):
        from werkzeug.datastructures import FileStorage

        csv_bytes = b"hash,category\nabc,good\n"
        fs = FileStorage(stream=io.BytesIO(csv_bytes), filename="labels.csv")
        with pytest.raises(ValueError, match="md5"):
            self._get_importer().run({"file": fs})

    def test_run_raises_on_empty_file(self):
        from werkzeug.datastructures import FileStorage

        fs = FileStorage(stream=io.BytesIO(b""), filename="labels.csv")
        with pytest.raises(ValueError):
            self._get_importer().run({"file": fs})

    def test_run_raises_when_no_file(self):
        with pytest.raises(ValueError):
            self._get_importer().run({"file": None})

    def test_run_cli_reads_file_path(self, tmp_path):
        p = tmp_path / "labels.csv"
        p.write_text("md5,label\nxyz789,bad\n")
        result = self._get_importer().run_cli({"file": str(p)})
        assert len(result) == 1
        assert result[0]["md5"] == "xyz789"

    def test_run_cli_raises_on_empty_path(self):
        with pytest.raises(ValueError, match="--file"):
            self._get_importer().run_cli({"file": ""})

    def test_run_ignores_extra_columns(self):
        from werkzeug.datastructures import FileStorage

        csv_bytes = b"md5,label,score,extra\nabc,good,0.9,whatever\n"
        fs = FileStorage(stream=io.BytesIO(csv_bytes), filename="labels.csv")
        result = self._get_importer().run({"file": fs})
        assert len(result) == 1
        assert result[0] == {"md5": "abc", "label": "good"}


# ---------------------------------------------------------------------------
# parse helpers (_parse_json_bytes, _parse_csv_bytes)
# ---------------------------------------------------------------------------


class TestParseHelpers:
    def test_parse_json_bytes_valid(self):
        from vtsearch.labels.importers.json_file import _parse_json_bytes

        raw = json.dumps({"labels": [{"md5": "a", "label": "good"}]}).encode()
        result = _parse_json_bytes(raw)
        assert result == [{"md5": "a", "label": "good"}]

    def test_parse_json_bytes_empty_labels(self):
        from vtsearch.labels.importers.json_file import _parse_json_bytes

        raw = json.dumps({"labels": []}).encode()
        result = _parse_json_bytes(raw)
        assert result == []

    def test_parse_json_bytes_non_dict_entries_filtered(self):
        from vtsearch.labels.importers.json_file import _parse_json_bytes

        raw = json.dumps({"labels": [{"md5": "a", "label": "good"}, "bad_entry", 42]}).encode()
        result = _parse_json_bytes(raw)
        assert len(result) == 1

    def test_parse_csv_bytes_valid(self):
        from vtsearch.labels.importers.csv_file import _parse_csv_bytes

        raw = b"md5,label\nabc,good\ndef,bad\n"
        result = _parse_csv_bytes(raw)
        assert len(result) == 2

    def test_parse_csv_bytes_case_insensitive_headers(self):
        from vtsearch.labels.importers.csv_file import _parse_csv_bytes

        raw = b"MD5,LABEL\nabc,good\n"
        result = _parse_csv_bytes(raw)
        assert len(result) == 1
        assert result[0]["md5"] == "abc"

    def test_parse_csv_bytes_skips_empty_md5(self):
        from vtsearch.labels.importers.csv_file import _parse_csv_bytes

        raw = b"md5,label\n,good\nabc,bad\n"
        result = _parse_csv_bytes(raw)
        # Empty md5 row is skipped
        assert len(result) == 1
        assert result[0]["md5"] == "abc"


# ---------------------------------------------------------------------------
# API – GET /api/label-importers
# ---------------------------------------------------------------------------


class TestGetLabelImportersEndpoint:
    def test_returns_200(self, client):
        res = client.get("/api/label-importers")
        assert res.status_code == 200

    def test_returns_list(self, client):
        res = client.get("/api/label-importers")
        data = res.get_json()
        assert isinstance(data, list)

    def test_contains_builtin_importers(self, client):
        res = client.get("/api/label-importers")
        names = {entry["name"] for entry in res.get_json()}
        assert "json_file" in names
        assert "csv_file" in names

    def test_each_entry_has_required_keys(self, client):
        res = client.get("/api/label-importers")
        for entry in res.get_json():
            assert "name" in entry
            assert "display_name" in entry
            assert "description" in entry
            assert "icon" in entry
            assert "fields" in entry


# ---------------------------------------------------------------------------
# API – POST /api/label-importers/import/<name>
# ---------------------------------------------------------------------------


class TestLabelImportEndpoint:
    def test_unknown_importer_returns_404(self, client):
        res = client.post("/api/label-importers/import/no_such_importer")
        assert res.status_code == 404
        assert "no_such_importer" in res.get_json()["error"]

    def test_json_importer_applies_good_label(self, client):
        md5 = app_module.clips[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "good"}]}).encode()
        data = {"file": (io.BytesIO(payload), "labels.json")}
        res = client.post(
            "/api/label-importers/import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 1
        assert result["skipped"] == 0
        assert 1 in app_module.good_votes

    def test_json_importer_applies_bad_label(self, client):
        md5 = app_module.clips[2]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "bad"}]}).encode()
        data = {"file": (io.BytesIO(payload), "labels.json")}
        res = client.post(
            "/api/label-importers/import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        assert 2 in app_module.bad_votes

    def test_json_importer_skips_unknown_md5(self, client):
        payload = json.dumps({"labels": [{"md5": "no_such_md5", "label": "good"}]}).encode()
        data = {"file": (io.BytesIO(payload), "labels.json")}
        res = client.post(
            "/api/label-importers/import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 0
        assert result["skipped"] == 1

    def test_json_importer_skips_invalid_label_value(self, client):
        md5 = app_module.clips[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "meh"}]}).encode()
        data = {"file": (io.BytesIO(payload), "labels.json")}
        res = client.post(
            "/api/label-importers/import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 0
        assert result["skipped"] == 1

    def test_csv_importer_applies_labels(self, client):
        md5_1 = app_module.clips[1]["md5"]
        md5_2 = app_module.clips[2]["md5"]
        csv_bytes = f"md5,label\n{md5_1},good\n{md5_2},bad\n".encode()
        data = {"file": (io.BytesIO(csv_bytes), "labels.csv")}
        res = client.post(
            "/api/label-importers/import/csv_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 2
        assert 1 in app_module.good_votes
        assert 2 in app_module.bad_votes

    def test_csv_importer_skips_unknown_md5(self, client):
        csv_bytes = b"md5,label\nunknown_hash,good\n"
        data = {"file": (io.BytesIO(csv_bytes), "labels.csv")}
        res = client.post(
            "/api/label-importers/import/csv_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 0
        assert result["skipped"] == 1

    def test_import_overrides_existing_label(self, client):
        app_module.good_votes[1] = None
        md5 = app_module.clips[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "bad"}]}).encode()
        data = {"file": (io.BytesIO(payload), "labels.json")}
        client.post(
            "/api/label-importers/import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert 1 not in app_module.good_votes
        assert 1 in app_module.bad_votes

    def test_import_response_has_message(self, client):
        payload = json.dumps({"labels": []}).encode()
        data = {"file": (io.BytesIO(payload), "labels.json")}
        res = client.post(
            "/api/label-importers/import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        assert "message" in res.get_json()

    def test_json_roundtrip_via_importer(self, client):
        """Export labels via the old route and re-import via label importer endpoint."""
        app_module.good_votes.update({k: None for k in [1, 3, 5]})
        app_module.bad_votes.update({k: None for k in [2, 4]})

        export_res = client.get("/api/labels/export")
        exported = export_res.get_json()

        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        raw = json.dumps(exported).encode()
        data = {"file": (io.BytesIO(raw), "labels.json")}
        res = client.post(
            "/api/label-importers/import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        result = res.get_json()
        assert result["applied"] == 5
        assert set(app_module.good_votes) == {1, 3, 5}
        assert set(app_module.bad_votes) == {2, 4}

    def test_multiple_clips_via_csv(self, client):
        lines = ["md5,label"]
        good_ids = [1, 2, 3]
        bad_ids = [4, 5]
        for cid in good_ids:
            lines.append(f"{app_module.clips[cid]['md5']},good")
        for cid in bad_ids:
            lines.append(f"{app_module.clips[cid]['md5']},bad")
        csv_bytes = "\n".join(lines).encode()
        data = {"file": (io.BytesIO(csv_bytes), "labels.csv")}
        res = client.post(
            "/api/label-importers/import/csv_file",
            data=data,
            content_type="multipart/form-data",
        )
        result = res.get_json()
        assert result["applied"] == 5
        assert set(app_module.good_votes) == {1, 2, 3}
        assert set(app_module.bad_votes) == {4, 5}
