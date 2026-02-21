"""Tests for the CLI autodetect feature (vtsearch/cli.py)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import app as app_module
from vtsearch.cli import (
    _build_results_dict,
    _detect_media_type,
    _run_exporter,
    run_autodetect,
    run_autodetect_with_importer,
)
from vtsearch.datasets.loader import export_dataset_to_file


# ---------------------------------------------------------------------------
# Helpers
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


_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Tests for run_autodetect()
# ---------------------------------------------------------------------------


class TestRunAutodetect:
    """Tests for the core run_autodetect function."""

    def test_returns_list(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        assert isinstance(hits, list)

    def test_hits_have_required_fields(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        for hit in hits:
            assert "id" in hit
            assert "filename" in hit
            assert "category" in hit
            assert "score" in hit

    def test_scores_in_valid_range(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        for hit in hits:
            assert 0.0 <= hit["score"] <= 1.0

    def test_hits_sorted_descending_by_score(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_all_hits_at_or_above_threshold(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])
        threshold = detector["threshold"]

        hits = run_autodetect(str(dataset_path), str(detector_path))
        for hit in hits:
            assert hit["score"] >= threshold - 1e-6  # float tolerance

    def test_good_clips_score_higher_than_bad(self, client, tmp_path):
        """Clips that were voted good should tend to score higher."""
        good_ids = [1, 2, 3]
        bad_ids = [18, 19, 20]
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, good_ids, bad_ids)

        # Score all clips by re-running with a threshold of 0 so every clip is included
        detector_data = json.loads(detector_path.read_text())
        detector_data["threshold"] = 0.0
        low_threshold_path = tmp_path / "detector_low.json"
        low_threshold_path.write_text(json.dumps(detector_data))

        all_hits = run_autodetect(str(dataset_path), str(low_threshold_path))
        all_score_map = {h["id"]: h["score"] for h in all_hits}

        avg_good = np.mean([all_score_map.get(i, 0) for i in good_ids])
        avg_bad = np.mean([all_score_map.get(i, 0) for i in bad_ids])
        assert avg_good > avg_bad

    def test_dataset_file_not_found(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            run_autodetect("/nonexistent/dataset.pkl", str(detector_path))

    def test_detector_file_not_found(self, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        with pytest.raises(FileNotFoundError, match="Detector file not found"):
            run_autodetect(str(dataset_path), "/nonexistent/detector.json")

    def test_empty_dataset_raises_error(self, client, tmp_path):
        empty_clips: dict = {}
        dataset_path = _make_dataset_file(tmp_path, empty_clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        with pytest.raises(ValueError, match="No clips loaded"):
            run_autodetect(str(dataset_path), str(detector_path))

    def test_detector_missing_weights_raises_error(self, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        bad_detector = tmp_path / "bad_detector.json"
        bad_detector.write_text(json.dumps({"threshold": 0.5}))

        with pytest.raises(ValueError, match="missing 'weights'"):
            run_autodetect(str(dataset_path), str(bad_detector))

    def test_detector_missing_threshold_raises_error(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        # Write detector with threshold removed
        del detector["threshold"]
        no_threshold_path = tmp_path / "no_threshold.json"
        no_threshold_path.write_text(json.dumps(detector))

        with pytest.raises(ValueError, match="missing 'threshold'"):
            run_autodetect(str(dataset_path), str(no_threshold_path))

    def test_hits_do_not_contain_embedding(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        for hit in hits:
            assert "embedding" not in hit

    def test_hits_do_not_contain_raw_media(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        for hit in hits:
            assert "wav_bytes" not in hit
            assert "image_bytes" not in hit
            assert "video_bytes" not in hit
            assert "text_content" not in hit

    def test_with_threshold_zero_returns_all_clips(self, client, tmp_path):
        """A threshold of 0 should return all clips since sigmoid output >= 0."""
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        detector["threshold"] = 0.0
        zero_path = tmp_path / "zero_threshold.json"
        zero_path.write_text(json.dumps(detector))

        hits = run_autodetect(str(dataset_path), str(zero_path))
        assert len(hits) == len(app_module.clips)

    def test_with_threshold_one_returns_few_or_none(self, client, tmp_path):
        """A threshold of 1.0 should return very few or no clips."""
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        detector["threshold"] = 1.0
        high_path = tmp_path / "high_threshold.json"
        high_path.write_text(json.dumps(detector))

        hits = run_autodetect(str(dataset_path), str(high_path))
        # With threshold=1.0 most clips should be excluded (sigmoid rarely reaches 1.0)
        assert len(hits) < len(app_module.clips)


# ---------------------------------------------------------------------------
# Tests for run_autodetect_with_importer()
# ---------------------------------------------------------------------------


class TestRunAutodetectWithImporter:
    """Tests for the importer-aware autodetect path."""

    def test_pickle_importer_returns_same_as_legacy(self, client, tmp_path):
        """Using --importer pickle should produce the same results as --dataset."""
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        legacy_hits = run_autodetect(str(dataset_path), str(detector_path))
        importer_hits = run_autodetect_with_importer("pickle", {"file": str(dataset_path)}, str(detector_path))

        assert len(legacy_hits) == len(importer_hits)
        for lh, ih in zip(legacy_hits, importer_hits):
            assert lh["id"] == ih["id"]
            assert lh["score"] == ih["score"]

    def test_pickle_importer_file_not_found(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            run_autodetect_with_importer("pickle", {"file": "/nonexistent.pkl"}, str(detector_path))

    def test_unknown_importer_raises_error(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])
        with pytest.raises(ValueError, match="Unknown importer"):
            run_autodetect_with_importer("nonexistent_importer", {}, str(detector_path))

    def test_missing_required_field_raises_error(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])
        with pytest.raises(ValueError, match="Missing required argument"):
            run_autodetect_with_importer("pickle", {}, str(detector_path))

    def test_folder_importer_nonexistent_path_raises(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])
        with pytest.raises(FileNotFoundError, match="Folder not found"):
            run_autodetect_with_importer(
                "folder",
                {"path": "/nonexistent/folder", "media_type": "sounds"},
                str(detector_path),
            )

    def test_folder_importer_file_instead_of_dir_raises(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])
        # Create a file (not a directory)
        fake_file = tmp_path / "not_a_dir.txt"
        fake_file.write_text("not a directory")
        with pytest.raises(NotADirectoryError, match="Not a directory"):
            run_autodetect_with_importer(
                "folder",
                {"path": str(fake_file), "media_type": "sounds"},
                str(detector_path),
            )

    def test_http_archive_invalid_url_raises(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])
        with pytest.raises(ValueError, match="Invalid URL"):
            run_autodetect_with_importer(
                "http_archive",
                {"url": "not-a-url", "media_type": "sounds"},
                str(detector_path),
            )


# ---------------------------------------------------------------------------
# Tests for importer CLI argument generation
# ---------------------------------------------------------------------------


class TestImporterCLIArguments:
    """Tests for add_cli_arguments and validate_cli_field_values."""

    def test_folder_importer_adds_expected_args(self):
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        imp = FolderDatasetImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--path", "/tmp/data", "--media-type", "images"])
        assert args.path == "/tmp/data"
        assert args.media_type == "images"

    def test_folder_importer_media_type_default(self):
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        imp = FolderDatasetImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--path", "/tmp/data"])
        assert args.media_type == "sounds"

    def test_folder_importer_rejects_invalid_media_type(self):
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        imp = FolderDatasetImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--path", "/tmp/data", "--media-type", "invalid"])

    def test_pickle_importer_adds_file_arg(self):
        from vtsearch.datasets.importers.pickle import PickleDatasetImporter

        imp = PickleDatasetImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--file", "/tmp/dataset.pkl"])
        assert args.file == "/tmp/dataset.pkl"

    def test_http_archive_importer_adds_expected_args(self):
        from vtsearch.datasets.importers.http_zip import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--url", "https://example.com/archive.zip", "--media-type", "images"])
        assert args.url == "https://example.com/archive.zip"
        assert args.media_type == "images"

    def test_validate_catches_missing_required_field(self):
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        imp = FolderDatasetImporter()
        with pytest.raises(ValueError, match="Missing required argument: --path"):
            imp.validate_cli_field_values({"media_type": "sounds"})

    def test_validate_passes_with_all_fields(self):
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        imp = FolderDatasetImporter()
        # Should not raise
        imp.validate_cli_field_values({"media_type": "sounds", "path": "/tmp/data"})


# ---------------------------------------------------------------------------
# Tests for CLI entry point (app.py --autodetect)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAutodetectCLI:
    """Tests for the command-line --autodetect flag via subprocess."""

    def test_autodetect_prints_output(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--dataset",
                str(dataset_path),
                "--detector",
                str(detector_path),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0
        assert "Predicted Good" in result.stdout or "No items predicted as Good" in result.stdout

    def test_autodetect_missing_dataset_flag(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        result = subprocess.run(
            [sys.executable, "app.py", "--autodetect", "--detector", str(detector_path)],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode != 0

    def test_autodetect_missing_detector_flag(self, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)

        result = subprocess.run(
            [sys.executable, "app.py", "--autodetect", "--dataset", str(dataset_path)],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode != 0

    def test_autodetect_nonexistent_dataset(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--dataset",
                "/tmp/nonexistent.pkl",
                "--detector",
                str(detector_path),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_autodetect_output_contains_names(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        # Use threshold 0 to ensure all clips appear
        detector["threshold"] = 0.0
        zero_path = tmp_path / "zero_threshold.json"
        zero_path.write_text(json.dumps(detector))

        result = subprocess.run(
            [sys.executable, "app.py", "--autodetect", "--dataset", str(dataset_path), "--detector", str(zero_path)],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0
        # Default gui exporter lists names, not scores
        assert "Predicted Good" in result.stdout

    def test_autodetect_no_hits_message(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        # Use threshold 1.0 to ensure no hits
        detector["threshold"] = 1.0
        high_path = tmp_path / "high_threshold.json"
        high_path.write_text(json.dumps(detector))

        result = subprocess.run(
            [sys.executable, "app.py", "--autodetect", "--dataset", str(dataset_path), "--detector", str(high_path)],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0
        assert "No items predicted as Good" in result.stdout


# ---------------------------------------------------------------------------
# Tests for CLI --importer flag via subprocess
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAutodetectImporterCLI:
    """Tests for the --autodetect --importer path via subprocess."""

    def test_importer_pickle_via_cli(self, client, tmp_path):
        """--importer pickle --file <path> should work like --dataset <path>."""
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--importer",
                "pickle",
                "--file",
                str(dataset_path),
                "--detector",
                str(detector_path),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0
        assert "Predicted Good" in result.stdout or "No items predicted as Good" in result.stdout

    def test_importer_unknown_name_fails(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--importer",
                "nonexistent_importer",
                "--detector",
                str(detector_path),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode != 0
        assert "Unknown importer" in result.stderr

    def test_importer_missing_detector_fails(self, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--importer",
                "pickle",
                "--file",
                str(dataset_path),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode != 0

    def test_importer_missing_required_field_fails(self, client, tmp_path):
        """Omitting a required importer field should produce an error."""
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--importer",
                "pickle",
                "--detector",
                str(detector_path),
                # --file intentionally omitted
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_importer_folder_nonexistent_path_fails(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--importer",
                "folder",
                "--path",
                "/nonexistent/folder",
                "--media-type",
                "sounds",
                "--detector",
                str(detector_path),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr


# ---------------------------------------------------------------------------
# Tests for exporter CLI argument generation
# ---------------------------------------------------------------------------


class TestExporterCLIArguments:
    """Tests for add_cli_arguments and validate_cli_field_values on exporters."""

    def test_file_exporter_adds_filepath_arg(self):
        from vtsearch.exporters.file import FileLabelsetExporter

        exp = FileLabelsetExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args(["--filepath", "/tmp/results.json"])
        assert args.filepath == "/tmp/results.json"

    def test_file_exporter_filepath_default(self):
        from vtsearch.exporters.file import FileLabelsetExporter

        exp = FileLabelsetExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args([])
        assert args.filepath == "autodetect_results.json"

    def test_email_smtp_exporter_adds_expected_args(self):
        from vtsearch.exporters.email_smtp import EmailLabelsetExporter

        exp = EmailLabelsetExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args(
            [
                "--to",
                "recipient@example.com",
                "--from-email",
                "sender@example.com",
                "--smtp-password",
                "secret",
            ]
        )
        assert args.to == "recipient@example.com"
        assert args.from_email == "sender@example.com"
        assert args.smtp_password == "secret"
        assert args.smtp_host == "smtp.gmail.com"
        assert args.smtp_port == "587"

    def test_email_smtp_exporter_custom_host_and_port(self):
        from vtsearch.exporters.email_smtp import EmailLabelsetExporter

        exp = EmailLabelsetExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        args = parser.parse_args(
            [
                "--to",
                "a@b.com",
                "--from-email",
                "c@d.com",
                "--smtp-password",
                "pw",
                "--smtp-host",
                "mail.example.com",
                "--smtp-port",
                "465",
            ]
        )
        assert args.smtp_host == "mail.example.com"
        assert args.smtp_port == "465"

    def test_gui_exporter_adds_no_args(self):
        from vtsearch.exporters.gui import DisplayLabelsetExporter

        exp = DisplayLabelsetExporter()
        parser = argparse.ArgumentParser()
        exp.add_cli_arguments(parser)

        # Should parse successfully with no extra args
        args = parser.parse_args([])
        assert not hasattr(args, "filepath")

    def test_validate_catches_missing_required_field(self):
        from vtsearch.exporters.email_smtp import EmailLabelsetExporter

        exp = EmailLabelsetExporter()
        with pytest.raises(ValueError, match="Missing required argument: --to"):
            exp.validate_cli_field_values({})

    def test_validate_passes_with_all_fields(self):
        from vtsearch.exporters.email_smtp import EmailLabelsetExporter

        exp = EmailLabelsetExporter()
        # Should not raise
        exp.validate_cli_field_values(
            {
                "to": "a@b.com",
                "from_email": "c@d.com",
                "smtp_password": "pw",
                "smtp_host": "smtp.gmail.com",
                "smtp_port": "587",
            }
        )

    def test_file_exporter_validate_passes(self):
        from vtsearch.exporters.file import FileLabelsetExporter

        exp = FileLabelsetExporter()
        exp.validate_cli_field_values({"filepath": "/tmp/out.json"})

    def test_gui_exporter_validate_passes_empty(self):
        from vtsearch.exporters.gui import DisplayLabelsetExporter

        exp = DisplayLabelsetExporter()
        # No required fields — empty dict is fine
        exp.validate_cli_field_values({})


# ---------------------------------------------------------------------------
# Tests for _build_results_dict and _detect_media_type
# ---------------------------------------------------------------------------


class TestBuildResultsDict:
    """Tests for the _build_results_dict helper."""

    def test_basic_structure(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        results = _build_results_dict(hits, str(detector_path), "audio")

        assert results["media_type"] == "audio"
        assert results["detectors_run"] == 1
        assert isinstance(results["results"], dict)
        assert len(results["results"]) == 1

    def test_detector_name_from_json(self, client, tmp_path):
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        # Write a detector with an explicit name field
        detector["name"] = "my_detector"
        named_path = tmp_path / "named_detector.json"
        named_path.write_text(json.dumps(detector))

        results = _build_results_dict([], str(named_path))
        assert "my_detector" in results["results"]

    def test_detector_name_falls_back_to_stem(self, client, tmp_path):
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2], [3, 4], name="bark_detector.json")

        # Remove name field if present
        detector.pop("name", None)
        detector_path.write_text(json.dumps(detector))

        results = _build_results_dict([], str(detector_path))
        det_names = list(results["results"].keys())
        assert len(det_names) == 1
        assert det_names[0] == "bark_detector"

    def test_hits_included_in_results(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        results = _build_results_dict(hits, str(detector_path))

        det_result = list(results["results"].values())[0]
        assert det_result["total_hits"] == len(hits)
        assert det_result["hits"] == hits

    def test_threshold_from_detector(self, client, tmp_path):
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2], [3, 4])

        results = _build_results_dict([], str(detector_path))
        det_result = list(results["results"].values())[0]
        assert det_result["threshold"] == detector["threshold"]

    def test_default_media_type_is_unknown(self, client, tmp_path):
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2], [3, 4])
        results = _build_results_dict([], str(detector_path))
        assert results["media_type"] == "unknown"


class TestDetectMediaType:
    """Tests for the _detect_media_type helper."""

    def test_returns_type_from_clips(self):
        clips_dict = {1: {"type": "audio"}, 2: {"type": "audio"}}
        assert _detect_media_type(clips_dict) == "audio"

    def test_returns_unknown_for_empty_clips(self):
        assert _detect_media_type({}) == "unknown"

    def test_returns_unknown_when_type_missing(self):
        clips_dict = {1: {"filename": "test.wav"}}
        assert _detect_media_type(clips_dict) == "unknown"


# ---------------------------------------------------------------------------
# Tests for _run_exporter
# ---------------------------------------------------------------------------


class TestRunExporter:
    """Tests for the _run_exporter function."""

    def test_file_exporter_creates_file(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        hits = run_autodetect(str(dataset_path), str(detector_path))
        results = _build_results_dict(hits, str(detector_path), "audio")

        output_file = tmp_path / "export_output.json"
        _run_exporter("file", {"filepath": str(output_file)}, results)

        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert saved["media_type"] == "audio"
        assert saved["detectors_run"] == 1

    def test_gui_exporter_prints_results(self, client, tmp_path, capsys):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, detector = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        # Use threshold 0 to ensure hits
        detector["threshold"] = 0.0
        zero_path = tmp_path / "zero_threshold.json"
        zero_path.write_text(json.dumps(detector))

        hits = run_autodetect(str(dataset_path), str(zero_path))
        results = _build_results_dict(hits, str(zero_path), "audio")

        _run_exporter("gui", {}, results)
        captured = capsys.readouterr()
        # gui exporter prints origins+names (no scores) or the confirmation message
        assert "Predicted Good" in captured.out or "Printed" in captured.out
        assert "score:" not in captured.out.lower()

    def test_unknown_exporter_raises_error(self):
        with pytest.raises(ValueError, match="Unknown exporter"):
            _run_exporter("nonexistent_exporter", {}, {})

    def test_missing_required_field_raises_error(self):
        with pytest.raises(ValueError, match="Missing required argument"):
            _run_exporter("email_smtp", {}, {})


# ---------------------------------------------------------------------------
# Tests for autodetect_main with --exporter
# ---------------------------------------------------------------------------


class TestAutodetectMainWithExporter:
    """Tests for the autodetect_main function with exporter support."""

    def test_file_exporter_via_function(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        output_file = tmp_path / "fn_export.json"
        from vtsearch.cli import autodetect_main

        autodetect_main(
            str(dataset_path),
            str(detector_path),
            exporter_name="file",
            exporter_field_values={"filepath": str(output_file)},
        )

        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert "results" in saved

    def test_no_exporter_uses_gui_default(self, client, tmp_path, capsys):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        from vtsearch.cli import autodetect_main

        autodetect_main(str(dataset_path), str(detector_path))

        captured = capsys.readouterr()
        # Default exporter is gui, which prints origins+names or "No items predicted"
        assert "Predicted Good" in captured.out or "No items predicted as Good" in captured.out
        # Should NOT contain score or category info (gui exporter strips those)
        assert "score:" not in captured.out.lower()


class TestAutodetectImporterMainWithExporter:
    """Tests for the autodetect_importer_main function with exporter support."""

    def test_pickle_importer_file_exporter(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        output_file = tmp_path / "importer_export.json"
        from vtsearch.cli import autodetect_importer_main

        autodetect_importer_main(
            "pickle",
            {"file": str(dataset_path)},
            str(detector_path),
            exporter_name="file",
            exporter_field_values={"filepath": str(output_file)},
        )

        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert "results" in saved
        assert saved["media_type"] == "audio"


# ---------------------------------------------------------------------------
# Tests for CLI --exporter flag via subprocess
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAutodetectExporterCLI:
    """Tests for the --autodetect --exporter path via subprocess."""

    def test_file_exporter_via_cli(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        output_file = tmp_path / "cli_export.json"
        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--dataset",
                str(dataset_path),
                "--detector",
                str(detector_path),
                "--exporter",
                "file",
                "--filepath",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert "results" in saved

    def test_file_exporter_with_importer_via_cli(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        output_file = tmp_path / "cli_imp_export.json"
        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--importer",
                "pickle",
                "--file",
                str(dataset_path),
                "--detector",
                str(detector_path),
                "--exporter",
                "file",
                "--filepath",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert "results" in saved

    def test_gui_exporter_via_cli(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--dataset",
                str(dataset_path),
                "--detector",
                str(detector_path),
                "--exporter",
                "gui",
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (
            "Predicted Good" in result.stdout
            or "No items predicted as Good" in result.stdout
            or "Printed" in result.stdout
        )
        # gui exporter should not include scores
        assert "score:" not in result.stdout.lower()

    def test_unknown_exporter_fails(self, client, tmp_path):
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--dataset",
                str(dataset_path),
                "--detector",
                str(detector_path),
                "--exporter",
                "nonexistent_exporter",
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode != 0
        assert "Unknown exporter" in result.stderr

    def test_missing_exporter_required_field_fails(self, client, tmp_path):
        """Omitting required email_smtp fields should produce an error."""
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--dataset",
                str(dataset_path),
                "--detector",
                str(detector_path),
                "--exporter",
                "email_smtp",
                # --to and other required fields intentionally omitted
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_file_exporter_output_contains_media_type(self, client, tmp_path):
        """File exporter output should include the media_type from the dataset."""
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        output_file = tmp_path / "media_type_test.json"
        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--dataset",
                str(dataset_path),
                "--detector",
                str(detector_path),
                "--exporter",
                "file",
                "--filepath",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        saved = json.loads(output_file.read_text())
        assert saved["media_type"] == "audio"

    def test_file_exporter_stdout_shows_confirmation(self, client, tmp_path):
        """The CLI should print the exporter's confirmation message."""
        dataset_path = _make_dataset_file(tmp_path, app_module.clips)
        detector_path, _ = _make_detector_file(tmp_path, client, [1, 2, 3], [18, 19, 20])

        output_file = tmp_path / "confirm_test.json"
        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--autodetect",
                "--dataset",
                str(dataset_path),
                "--detector",
                str(detector_path),
                "--exporter",
                "file",
                "--filepath",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0
        assert "Saved" in result.stdout
