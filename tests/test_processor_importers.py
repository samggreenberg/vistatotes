"""Tests for the Processor Importer abstraction.

Covers:
- ProcessorImporterField and ProcessorImporter base classes
- Auto-discovery registry (list_processor_importers, get_processor_importer)
- Built-in importers: detector_file, label_file
- Flask API routes: GET /api/processor-importers, POST /api/processor-importers/import/<name>
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import app as app_module  # noqa: F401 — triggers conftest clip init


# ---------------------------------------------------------------------------
# ProcessorImporterField
# ---------------------------------------------------------------------------


class TestProcessorImporterField:
    def test_to_dict_contains_required_keys(self):
        from vtsearch.processors.importers.base import ProcessorImporterField

        f = ProcessorImporterField(key="file", label="My File", field_type="file")
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
        from vtsearch.processors.importers.base import ProcessorImporterField

        f = ProcessorImporterField(key="x", label="X", field_type="text")
        assert f.required is True
        assert f.default == ""
        assert f.placeholder == ""
        assert f.options == []
        assert f.description == ""
        assert f.accept == ""

    def test_custom_values(self):
        from vtsearch.processors.importers.base import ProcessorImporterField

        f = ProcessorImporterField(
            key="mode",
            label="Mode",
            field_type="select",
            options=["a", "b"],
            default="a",
            required=False,
            description="Pick one",
            placeholder="Choose\u2026",
        )
        d = f.to_dict()
        assert d["options"] == ["a", "b"]
        assert d["default"] == "a"
        assert d["required"] is False


# ---------------------------------------------------------------------------
# ProcessorImporter base class
# ---------------------------------------------------------------------------


class TestProcessorImporterBase:
    def _make_minimal(self):
        from vtsearch.processors.importers.base import ProcessorImporter

        class Minimal(ProcessorImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "A minimal processor importer."
            fields = []

            def run(self, field_values):
                return {"media_type": "audio", "weights": {"0.weight": [[1]]}, "threshold": 0.5}

        return Minimal()

    def test_run_raises_not_implemented_when_not_overridden(self):
        from vtsearch.processors.importers.base import ProcessorImporter

        imp = ProcessorImporter()
        with pytest.raises(NotImplementedError):
            imp.run({})

    def test_to_dict_contains_standard_keys(self):
        imp = self._make_minimal()
        d = imp.to_dict()
        assert d["name"] == "minimal"
        assert d["display_name"] == "Minimal"
        assert d["description"] == "A minimal processor importer."
        assert "icon" in d
        assert "fields" in d

    def test_default_icon(self):
        from vtsearch.processors.importers.base import ProcessorImporter

        assert ProcessorImporter.icon == "\U0001f9e9"

    def test_custom_icon_in_to_dict(self):
        from vtsearch.processors.importers.base import ProcessorImporter

        class Custom(ProcessorImporter):
            name = "c"
            display_name = "C"
            description = "C"
            icon = "\U0001f4c4"
            fields = []

            def run(self, field_values):
                return {}

        assert Custom().to_dict()["icon"] == "\U0001f4c4"

    def test_validate_cli_field_values_raises_on_missing_required(self):
        from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField

        class Imp(ProcessorImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [ProcessorImporterField("filepath", "File", "text", required=True)]

            def run(self, field_values):
                return {}

        imp = Imp()
        with pytest.raises(ValueError, match="--filepath"):
            imp.validate_cli_field_values({})

    def test_validate_cli_field_values_passes_when_provided(self):
        from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField

        class Imp(ProcessorImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [ProcessorImporterField("filepath", "File", "text", required=True)]

            def run(self, field_values):
                return {}

        imp = Imp()
        imp.validate_cli_field_values({"filepath": "/some/path"})  # no raise

    def test_run_cli_delegates_to_run(self):
        imp = self._make_minimal()
        result = imp.run_cli({})
        assert result["media_type"] == "audio"
        assert result["weights"] == {"0.weight": [[1]]}

    def test_add_cli_arguments_adds_text_field(self):
        import argparse

        from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField

        class Imp(ProcessorImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [ProcessorImporterField("server", "Server", "text", description="DB host")]

            def run(self, field_values):
                return {}

        parser = argparse.ArgumentParser()
        Imp().add_cli_arguments(parser)
        args = parser.parse_args(["--server", "localhost"])
        assert args.server == "localhost"

    def test_add_cli_arguments_select_adds_choices(self):
        import argparse

        from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField

        class Imp(ProcessorImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [ProcessorImporterField("mode", "Mode", "select", options=["a", "b"], default="a")]

            def run(self, field_values):
                return {}

        parser = argparse.ArgumentParser()
        Imp().add_cli_arguments(parser)
        args = parser.parse_args([])  # uses default
        assert args.mode == "a"
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestProcessorImporterRegistry:
    def test_list_processor_importers_returns_builtins(self):
        from vtsearch.processors.importers import list_processor_importers

        names = {imp.name for imp in list_processor_importers()}
        assert "detector_file" in names
        assert "label_file" in names
        assert "csv_label_file" in names

    def test_get_processor_importer_known(self):
        from vtsearch.processors.importers import get_processor_importer

        for name in ("detector_file", "label_file", "csv_label_file"):
            imp = get_processor_importer(name)
            assert imp is not None, f"Processor importer '{name}' not found"
            assert imp.name == name

    def test_get_processor_importer_unknown_returns_none(self):
        from vtsearch.processors.importers import get_processor_importer

        assert get_processor_importer("no_such_importer") is None

    def test_each_importer_has_display_name_and_icon(self):
        from vtsearch.processors.importers import list_processor_importers

        for imp in list_processor_importers():
            assert imp.display_name, f"{imp.name} missing display_name"
            assert imp.icon, f"{imp.name} missing icon"
            assert imp.description, f"{imp.name} missing description"

    def test_each_importer_fields_are_valid(self):
        from vtsearch.processors.importers import list_processor_importers

        valid_types = ("file", "text", "password", "select")
        for imp in list_processor_importers():
            for f in imp.fields:
                assert f.key, f"{imp.name} has a field without a key"
                assert f.label, f"{imp.name} field '{f.key}' has no label"
                assert f.field_type in valid_types, (
                    f"{imp.name} field '{f.key}' has unknown type '{f.field_type}'"
                )


# ---------------------------------------------------------------------------
# Detector file importer
# ---------------------------------------------------------------------------


class TestDetectorFileImporter:
    def _get_importer(self):
        from vtsearch.processors.importers.detector_file import PROCESSOR_IMPORTER

        return PROCESSOR_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "detector_file"

    def test_display_name(self):
        assert "detector" in self._get_importer().display_name.lower()

    def test_icon(self):
        assert self._get_importer().icon

    def test_has_file_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "file" in fields
        assert fields["file"].field_type == "file"

    def test_run_with_file_storage(self):
        from werkzeug.datastructures import FileStorage

        payload = {
            "weights": {"0.weight": [[1.0, 2.0]], "0.bias": [0.5]},
            "threshold": 0.75,
            "media_type": "image",
        }
        raw = json.dumps(payload).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="detector.json", content_type="application/json")
        result = self._get_importer().run({"file": fs})
        assert result["media_type"] == "image"
        assert result["weights"] == payload["weights"]
        assert result["threshold"] == 0.75

    def test_run_defaults_media_type_to_audio(self):
        from werkzeug.datastructures import FileStorage

        payload = {"weights": {"0.weight": [[1.0]]}, "threshold": 0.5}
        raw = json.dumps(payload).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="detector.json")
        result = self._get_importer().run({"file": fs})
        assert result["media_type"] == "audio"

    def test_run_includes_suggested_name(self):
        from werkzeug.datastructures import FileStorage

        payload = {"weights": {"0.weight": [[1.0]]}, "threshold": 0.5, "name": "my detector"}
        raw = json.dumps(payload).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="detector.json")
        result = self._get_importer().run({"file": fs})
        assert result["name"] == "my detector"

    def test_run_raises_on_invalid_json(self):
        from werkzeug.datastructures import FileStorage

        fs = FileStorage(stream=io.BytesIO(b"not json"), filename="bad.json")
        with pytest.raises(ValueError, match="JSON"):
            self._get_importer().run({"file": fs})

    def test_run_raises_on_missing_weights(self):
        from werkzeug.datastructures import FileStorage

        raw = json.dumps({"threshold": 0.5}).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="bad.json")
        with pytest.raises(ValueError, match="weights"):
            self._get_importer().run({"file": fs})

    def test_run_raises_when_no_file(self):
        with pytest.raises(ValueError):
            self._get_importer().run({"file": None})

    def test_run_cli_reads_file_path(self, tmp_path):
        payload = {"weights": {"0.weight": [[1.0]], "0.bias": [0.1]}, "threshold": 0.6}
        p = tmp_path / "detector.json"
        p.write_text(json.dumps(payload))
        result = self._get_importer().run_cli({"file": str(p)})
        assert result["threshold"] == 0.6
        assert result["weights"] == payload["weights"]

    def test_run_cli_raises_on_empty_path(self):
        with pytest.raises(ValueError, match="--file"):
            self._get_importer().run_cli({"file": ""})


# ---------------------------------------------------------------------------
# Label file importer (mocked embedding/training)
# ---------------------------------------------------------------------------


class TestLabelFileImporter:
    def _get_importer(self):
        from vtsearch.processors.importers.label_file import PROCESSOR_IMPORTER

        return PROCESSOR_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "label_file"

    def test_display_name(self):
        assert "label" in self._get_importer().display_name.lower()

    def test_icon(self):
        assert self._get_importer().icon

    def test_has_file_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "file" in fields
        assert fields["file"].field_type == "file"

    def test_has_media_type_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "media_type" in fields
        assert fields["media_type"].field_type == "select"
        assert fields["media_type"].required is False

    def test_run_raises_when_no_file(self):
        with pytest.raises(ValueError):
            self._get_importer().run({"file": None})

    def test_run_raises_on_invalid_json(self):
        from werkzeug.datastructures import FileStorage

        fs = FileStorage(stream=io.BytesIO(b"not json"), filename="labels.json")
        with pytest.raises(ValueError, match="JSON"):
            self._get_importer().run({"file": fs})

    def test_run_raises_on_empty_labels(self):
        from werkzeug.datastructures import FileStorage

        raw = json.dumps({"labels": []}).encode()
        fs = FileStorage(stream=io.BytesIO(raw), filename="labels.json")
        with pytest.raises(ValueError, match="No labels"):
            self._get_importer().run({"file": fs})

    def test_run_cli_raises_on_empty_path(self):
        with pytest.raises(ValueError, match="--file"):
            self._get_importer().run_cli({"file": ""})

    def test_media_type_for_path(self):
        from vtsearch.processors.importers.label_file import _media_type_for_path

        assert _media_type_for_path(Path("test.wav")) == "audio"
        assert _media_type_for_path(Path("test.mp3")) == "audio"
        assert _media_type_for_path(Path("test.jpg")) == "image"
        assert _media_type_for_path(Path("test.png")) == "image"
        assert _media_type_for_path(Path("test.mp4")) == "video"
        assert _media_type_for_path(Path("test.txt")) == "paragraph"
        assert _media_type_for_path(Path("test.xyz")) is None


# ---------------------------------------------------------------------------
# API – GET /api/processor-importers
# ---------------------------------------------------------------------------


class TestGetProcessorImportersEndpoint:
    def test_returns_200(self, client):
        res = client.get("/api/processor-importers")
        assert res.status_code == 200

    def test_returns_list(self, client):
        res = client.get("/api/processor-importers")
        data = res.get_json()
        assert isinstance(data, list)

    def test_contains_builtin_importers(self, client):
        res = client.get("/api/processor-importers")
        names = {entry["name"] for entry in res.get_json()}
        assert "detector_file" in names
        assert "label_file" in names
        assert "csv_label_file" in names

    def test_each_entry_has_required_keys(self, client):
        res = client.get("/api/processor-importers")
        for entry in res.get_json():
            assert "name" in entry
            assert "display_name" in entry
            assert "description" in entry
            assert "icon" in entry
            assert "fields" in entry


# ---------------------------------------------------------------------------
# API – POST /api/processor-importers/import/<name>
# ---------------------------------------------------------------------------


class TestProcessorImportEndpoint:
    def test_unknown_importer_returns_404(self, client):
        res = client.post("/api/processor-importers/import/no_such_importer")
        assert res.status_code == 404
        assert "no_such_importer" in res.get_json()["error"]

    def test_missing_name_returns_400(self, client):
        payload = {
            "weights": {"0.weight": [[1.0]], "0.bias": [0.5]},
            "threshold": 0.5,
        }
        raw = json.dumps(payload).encode()
        data = {"file": (io.BytesIO(raw), "detector.json")}
        res = client.post(
            "/api/processor-importers/import/detector_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 400
        assert "name" in res.get_json()["error"].lower()

    def test_detector_file_imports_and_saves(self, client):
        from vtsearch.utils import favorite_detectors

        payload = {
            "weights": {"0.weight": [[1.0, 2.0]], "0.bias": [0.5]},
            "threshold": 0.75,
            "media_type": "image",
        }
        raw = json.dumps(payload).encode()
        data = {
            "file": (io.BytesIO(raw), "detector.json"),
            "name": "test_detector",
        }
        res = client.post(
            "/api/processor-importers/import/detector_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["success"] is True
        assert result["name"] == "test_detector"
        assert result["media_type"] == "image"
        assert "test_detector" in favorite_detectors

        # Clean up
        favorite_detectors.pop("test_detector", None)

    def test_detector_file_defaults_to_audio(self, client):
        from vtsearch.utils import favorite_detectors

        payload = {"weights": {"0.weight": [[1.0]]}, "threshold": 0.5}
        raw = json.dumps(payload).encode()
        data = {
            "file": (io.BytesIO(raw), "detector.json"),
            "name": "audio_det",
        }
        res = client.post(
            "/api/processor-importers/import/detector_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        assert res.get_json()["media_type"] == "audio"

        # Clean up
        favorite_detectors.pop("audio_det", None)

    def test_detector_file_invalid_json_returns_400(self, client):
        data = {
            "file": (io.BytesIO(b"not json"), "bad.json"),
            "name": "bad_det",
        }
        res = client.post(
            "/api/processor-importers/import/detector_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 400
        assert "json" in res.get_json()["error"].lower()

    def test_detector_file_missing_weights_returns_400(self, client):
        raw = json.dumps({"threshold": 0.5}).encode()
        data = {
            "file": (io.BytesIO(raw), "bad.json"),
            "name": "bad_det",
        }
        res = client.post(
            "/api/processor-importers/import/detector_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 400


# ---------------------------------------------------------------------------
# CSV label file importer
# ---------------------------------------------------------------------------


class TestCsvLabelFileImporter:
    def _get_importer(self):
        from vtsearch.processors.importers.csv_label_file import PROCESSOR_IMPORTER

        return PROCESSOR_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "csv_label_file"

    def test_display_name(self):
        assert "label" in self._get_importer().display_name.lower()
        assert ".csv" in self._get_importer().display_name.lower()

    def test_icon(self):
        assert self._get_importer().icon

    def test_has_file_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "file" in fields
        assert fields["file"].field_type == "file"
        assert ".csv" in fields["file"].accept

    def test_has_media_type_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "media_type" in fields
        assert fields["media_type"].field_type == "select"
        assert fields["media_type"].required is False

    def test_run_raises_when_no_file(self):
        with pytest.raises(ValueError):
            self._get_importer().run({"file": None})

    def test_run_raises_on_invalid_csv(self):
        from werkzeug.datastructures import FileStorage

        # CSV with wrong headers
        raw = b"foo,bar\na,b\n"
        fs = FileStorage(stream=io.BytesIO(raw), filename="labels.csv")
        with pytest.raises(ValueError, match="path"):
            self._get_importer().run({"file": fs})

    def test_run_raises_on_empty_csv(self):
        from werkzeug.datastructures import FileStorage

        raw = b"path,label\n"
        fs = FileStorage(stream=io.BytesIO(raw), filename="labels.csv")
        with pytest.raises(ValueError, match="No labels"):
            self._get_importer().run({"file": fs})

    def test_run_cli_raises_on_empty_path(self):
        with pytest.raises(ValueError, match="--file"):
            self._get_importer().run_cli({"file": ""})

    def test_parse_csv_bytes(self):
        from vtsearch.processors.importers.csv_label_file import _parse_csv_bytes

        raw = b"path,label\n/data/dog.wav,good\n/data/silence.wav,bad\n"
        entries = _parse_csv_bytes(raw)
        assert len(entries) == 2
        assert entries[0]["path"] == "/data/dog.wav"
        assert entries[0]["label"] == "good"
        assert entries[1]["path"] == "/data/silence.wav"
        assert entries[1]["label"] == "bad"

    def test_parse_csv_bytes_file_column(self):
        from vtsearch.processors.importers.csv_label_file import _parse_csv_bytes

        raw = b"file,label\n/data/dog.wav,good\n"
        entries = _parse_csv_bytes(raw)
        assert len(entries) == 1
        assert entries[0]["path"] == "/data/dog.wav"

    def test_parse_csv_bytes_filename_column(self):
        from vtsearch.processors.importers.csv_label_file import _parse_csv_bytes

        raw = b"filename,label\n/data/dog.wav,good\n"
        entries = _parse_csv_bytes(raw)
        assert len(entries) == 1
        assert entries[0]["path"] == "/data/dog.wav"

    def test_parse_csv_bytes_missing_label_column(self):
        from vtsearch.processors.importers.csv_label_file import _parse_csv_bytes

        with pytest.raises(ValueError, match="label"):
            _parse_csv_bytes(b"path,status\n/data/dog.wav,good\n")

    def test_parse_csv_bytes_missing_path_column(self):
        from vtsearch.processors.importers.csv_label_file import _parse_csv_bytes

        with pytest.raises(ValueError, match="path"):
            _parse_csv_bytes(b"md5,label\nabc,good\n")

    def test_registry_contains_csv_label_file(self):
        from vtsearch.processors.importers import list_processor_importers

        names = {imp.name for imp in list_processor_importers()}
        assert "csv_label_file" in names

    def test_api_lists_csv_label_file(self, client):
        res = client.get("/api/processor-importers")
        assert res.status_code == 200
        names = {entry["name"] for entry in res.get_json()}
        assert "csv_label_file" in names


# ---------------------------------------------------------------------------
# API – POST /api/favorite-detectors/from-label-import/<name>
# ---------------------------------------------------------------------------


class TestFromLabelImportEndpoint:
    def test_unknown_importer_returns_404(self, client):
        res = client.post("/api/favorite-detectors/from-label-import/no_such_importer")
        assert res.status_code == 404

    def test_missing_name_returns_400(self, client):
        raw = json.dumps({"labels": [{"md5": "abc", "label": "good"}]}).encode()
        data = {"file": (io.BytesIO(raw), "labels.json")}
        res = client.post(
            "/api/favorite-detectors/from-label-import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 400
        assert "name" in res.get_json()["error"].lower()

    def test_trains_from_matched_clips(self, client):
        from vtsearch.utils import clips, favorite_detectors

        # Build labels from actual loaded clip md5s
        md5s = []
        for cid in sorted(clips.keys()):
            md5s.append(clips[cid].get("md5", ""))
        if len(md5s) < 2:
            pytest.skip("Need at least 2 clips for this test")

        labels_data = {"labels": []}
        for i, md5 in enumerate(md5s):
            if not md5:
                continue
            labels_data["labels"].append({
                "md5": md5,
                "label": "good" if i % 2 == 0 else "bad",
            })

        if len(labels_data["labels"]) < 2:
            pytest.skip("Need at least 2 clips with md5 for this test")

        # Ensure we have at least one good and one bad
        has_good = any(e["label"] == "good" for e in labels_data["labels"])
        has_bad = any(e["label"] == "bad" for e in labels_data["labels"])
        if not (has_good and has_bad):
            pytest.skip("Need at least one good and one bad label")

        raw = json.dumps(labels_data).encode()
        data = {
            "file": (io.BytesIO(raw), "labels.json"),
            "name": "from_label_import_test",
        }
        res = client.post(
            "/api/favorite-detectors/from-label-import/json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["success"] is True
        assert result["name"] == "from_label_import_test"
        assert result["loaded"] >= 2
        assert "from_label_import_test" in favorite_detectors

        # Clean up
        favorite_detectors.pop("from_label_import_test", None)

    def test_no_clips_returns_400(self, client):
        from vtsearch.utils import clips

        saved = dict(clips)
        clips.clear()
        try:
            raw = json.dumps({"labels": [{"md5": "abc", "label": "good"}]}).encode()
            data = {
                "file": (io.BytesIO(raw), "labels.json"),
                "name": "test",
            }
            res = client.post(
                "/api/favorite-detectors/from-label-import/json_file",
                data=data,
                content_type="multipart/form-data",
            )
            assert res.status_code == 400
            assert "no clips" in res.get_json()["error"].lower()
        finally:
            clips.update(saved)


