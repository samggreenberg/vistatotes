"""The CLI resolves thin mode from the importer's ``reference_files`` field.

``server_folder`` and ``server_files`` declare a "Reference files in place"
checkbox, and :meth:`~vtscore.plugins.PluginBase.add_cli_arguments` already
turned it into ``--reference-files`` / ``--no-reference-files``.  The CLI used
to leave that value inert in ``field_values`` and force ``thin=True`` on every
import instead, which broke two things at once (issue #3556):

* the flag did nothing, so reference mode could not be turned off; and
* thin swaps ``media_bytes`` for a path reference, so an item whose bytes are
  not re-readable from outside the source lost its only copy.  It could then
  not be embedded, so it was silently skipped at scoring - and since the
  calibrated threshold is fitted on the haystack being scored, the smaller
  surviving population moved the cut too.  The same dataset and detector gave
  different hits *and* a lower threshold in the CLI than in the GUI.

The GUI's resolution is the reference implementation: ``load_pipeline`` pops
the field out of the values and passes it as ``thin=``.
"""

from __future__ import annotations

from typing import Any

import pytest

from vtscore.cli import _load_importer_whole, _reference_files_choice


class TestReferenceFilesChoice:
    def test_absent_field_means_no_reference_mode(self):
        assert _reference_files_choice({}) is False

    @pytest.mark.parametrize("value", [True, "true", "True", " TRUE "])
    def test_truthy_shapes_enable_it(self, value):
        assert _reference_files_choice({"reference_files": value}) is True

    @pytest.mark.parametrize("value", [False, "false", "", None, "no"])
    def test_everything_else_disables_it(self, value):
        assert _reference_files_choice({"reference_files": value}) is False

    def test_the_key_is_popped_not_read(self):
        """``run`` takes thin as a parameter, so the key must not be forwarded."""
        field_values: dict[str, Any] = {"path": "/data", "reference_files": True}
        _reference_files_choice(field_values)
        assert field_values == {"path": "/data"}


class _RecordingImporter:
    """Minimal importer that records the ``thin`` it was handed."""

    name = "recorder"

    def __init__(self) -> None:
        self.thin: bool | None = None
        self.field_values: dict[str, Any] | None = None

    def validate_cli_field_values(self, field_values: dict[str, Any]) -> None:
        return None

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        self.thin = thin
        self.field_values = dict(field_values)
        medias[1] = {"id": 1}


class TestLoaderThreading:
    @pytest.mark.parametrize(
        ("field_values", "expected"),
        [({}, False), ({"reference_files": False}, False), ({"reference_files": True}, True)],
    )
    def test_choice_reaches_the_importer_as_thin(self, monkeypatch, field_values, expected):
        importer = _RecordingImporter()
        monkeypatch.setattr("vtscore.datasets.importers.get_importer", lambda _n: importer)
        list(_load_importer_whole("recorder", dict(field_values)))
        assert importer.thin is expected

    def test_the_field_is_not_forwarded_to_the_importer(self, monkeypatch):
        importer = _RecordingImporter()
        monkeypatch.setattr("vtscore.datasets.importers.get_importer", lambda _n: importer)
        list(_load_importer_whole("recorder", {"path": "/data", "reference_files": True}))
        assert importer.field_values == {"path": "/data"}
