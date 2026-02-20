"""Tests for the CLI autodetect feature (vistatotes/cli.py)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import app as app_module
from vistatotes.cli import run_autodetect, run_autodetect_with_importer
from vistatotes.datasets.loader import export_dataset_to_file


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
        from vistatotes.datasets.importers.folder import FolderImporter

        imp = FolderImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--path", "/tmp/data", "--media-type", "images"])
        assert args.path == "/tmp/data"
        assert args.media_type == "images"

    def test_folder_importer_media_type_default(self):
        from vistatotes.datasets.importers.folder import FolderImporter

        imp = FolderImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--path", "/tmp/data"])
        assert args.media_type == "sounds"

    def test_folder_importer_rejects_invalid_media_type(self):
        from vistatotes.datasets.importers.folder import FolderImporter

        imp = FolderImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--path", "/tmp/data", "--media-type", "invalid"])

    def test_pickle_importer_adds_file_arg(self):
        from vistatotes.datasets.importers.pickle import PickleImporter

        imp = PickleImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--file", "/tmp/dataset.pkl"])
        assert args.file == "/tmp/dataset.pkl"

    def test_http_archive_importer_adds_expected_args(self):
        from vistatotes.datasets.importers.http_zip import HttpArchiveImporter

        imp = HttpArchiveImporter()
        parser = argparse.ArgumentParser()
        imp.add_cli_arguments(parser)

        args = parser.parse_args(["--url", "https://example.com/archive.zip", "--media-type", "images"])
        assert args.url == "https://example.com/archive.zip"
        assert args.media_type == "images"

    def test_validate_catches_missing_required_field(self):
        from vistatotes.datasets.importers.folder import FolderImporter

        imp = FolderImporter()
        with pytest.raises(ValueError, match="Missing required argument: --path"):
            imp.validate_cli_field_values({"media_type": "sounds"})

    def test_validate_passes_with_all_fields(self):
        from vistatotes.datasets.importers.folder import FolderImporter

        imp = FolderImporter()
        # Should not raise
        imp.validate_cli_field_values({"media_type": "sounds", "path": "/tmp/data"})


# ---------------------------------------------------------------------------
# Tests for CLI entry point (app.py --autodetect)
# ---------------------------------------------------------------------------


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

    def test_autodetect_output_contains_filenames(self, client, tmp_path):
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
        assert "score:" in result.stdout

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
