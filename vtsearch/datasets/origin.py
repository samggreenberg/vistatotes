"""Origin tracking for dataset elements.

An :class:`Origin` records *where* a data element came from — the importer
that produced it and the parameters that were used.  Each element in a
dataset carries its own ``Origin`` so that:

* Future data from a different source can be added to the same dataset.
* Exported label sets can be re-imported and matched back to their source.
* Provenance is preserved when datasets are saved and reloaded.

Examples::

    >>> Origin("folder", {"path": "/data/audio", "media_type": "sounds"})
    Origin(importer='folder', params={'path': '/data/audio', 'media_type': 'sounds'})

    >>> Origin("http_archive", {"url": "https://example.com/data.zip"}).display()
    'http_archive(https://example.com/data.zip)'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Origin:
    """Identifies the source of one or more data elements.

    Attributes:
        importer: The name of the importer that produced the elements, e.g.
            ``"folder"``, ``"http_archive"``, ``"demo"``.
        params: A dict of the identifying parameters for the import, e.g.
            ``{"path": "/data/audio", "media_type": "sounds"}``.  File-upload
            parameters are excluded (they cannot be reconstructed from a
            string).
    """

    importer: str
    params: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON or pickle."""
        return {"importer": self.importer, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Origin:
        """Reconstruct an ``Origin`` from a dict produced by :meth:`to_dict`.

        Args:
            d: A dict with ``"importer"`` (str) and optional ``"params"``
                (dict) keys.

        Returns:
            A new :class:`Origin` instance.
        """
        return cls(importer=d["importer"], params=d.get("params", {}))

    def display(self) -> str:
        """Return a short human-readable representation.

        Examples: ``"folder(/data/audio)"``, ``"demo(esc50_animals)"``.
        """
        if self.params:
            first_val = next(iter(self.params.values()))
            return f"{self.importer}({first_val})"
        return self.importer

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Origin):
            return NotImplemented
        return self.importer == other.importer and self.params == other.params

    def __hash__(self) -> int:
        return hash((self.importer, tuple(sorted(self.params.items()))))
