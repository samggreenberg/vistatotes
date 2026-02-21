"""Pickle-file importer \u2013 loads a previously exported ``.pkl`` dataset.

No additional pip packages are required; everything needed is already in
the core requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import DATA_DIR
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import load_dataset_from_pickle


def _get_progress():
    from vtsearch.utils import update_progress

    return update_progress


class PickleImporter(DatasetImporter):
    """Load a dataset from a ``.pkl`` file exported by VTSearch.

    The user picks the file via the browser's file-upload input.  The file
    is streamed to a temporary path on the server, deserialized, and then
    the temporary file is deleted.

    If the pickle file contains embedded ``creation_info`` (i.e. it was
    exported from a dataset that recorded its provenance), that info is
    restored into the global state so the provenance chain is preserved.
    """

    name = "pickle"
    display_name = "Pickle File"
    description = "Load a previously exported .pkl dataset file."
    fields = [
        ImporterField(
            key="file",
            label="Dataset File",
            field_type="file",
            description="A .pkl file that was exported from VTSearch.",
            accept=".pkl",
        ),
    ]

    def run(self, field_values: dict, clips: dict) -> None:
        from vtsearch.utils import set_dataset_creation_info

        file_obj = field_values["file"]  # werkzeug FileStorage
        progress = _get_progress()
        progress("loading", "Loading dataset from file...", 0, 0)
        temp_path = DATA_DIR / "temp_upload.pkl"
        DATA_DIR.mkdir(exist_ok=True)
        file_obj.save(temp_path)
        try:
            creation_info = load_dataset_from_pickle(temp_path, clips)
        finally:
            temp_path.unlink(missing_ok=True)
        if creation_info:
            set_dataset_creation_info(creation_info)
        progress("idle", f"Loaded {len(clips)} clips from file")

    def run_cli(self, field_values: dict[str, Any], clips: dict) -> None:
        """Load from a pickle file path (string) instead of FileStorage."""
        from vtsearch.utils import set_dataset_creation_info

        file_path = Path(field_values["file"])
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        creation_info = load_dataset_from_pickle(file_path, clips)
        if creation_info:
            set_dataset_creation_info(creation_info)


IMPORTER = PickleImporter()
