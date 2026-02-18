"""Media type registry.

All built-in media types are registered at the bottom of this module.
Third-party or project-specific types can be added by calling
:func:`register` after importing this module::

    from vistatotes.media import register
    from mypackage.media_type import SourceCodeMediaType

    register(SourceCodeMediaType())

The new type will then be picked up automatically by model initialisation,
dataset loading, HTTP routing, and the demo-dataset listing.
"""

from __future__ import annotations

from vistatotes.media.base import DemoDataset, MediaType

_registry: dict[str, "MediaType"] = {}


def register(media_type: "MediaType") -> None:
    """Add *media_type* to the registry, keyed by :attr:`~MediaType.type_id`."""
    _registry[media_type.type_id] = media_type


def get(type_id: str) -> "MediaType":
    """Return the :class:`MediaType` registered under *type_id*.

    Raises :class:`KeyError` if *type_id* is not registered.
    """
    if type_id not in _registry:
        raise KeyError(f"Unknown media type: {type_id!r}")
    return _registry[type_id]


def get_by_folder_name(folder_name: str) -> "MediaType":
    """Return the :class:`MediaType` whose :attr:`~MediaType.folder_import_name`
    matches *folder_name*.

    Raises :class:`KeyError` if no registered type has that folder name.
    """
    for mt in _registry.values():
        if mt.folder_import_name == folder_name:
            return mt
    raise KeyError(f"No media type with folder_import_name: {folder_name!r}")


def all_types() -> list["MediaType"]:
    """Return all registered :class:`MediaType` instances."""
    return list(_registry.values())


def all_demo_datasets() -> dict:
    """Return a flat ``{dataset_id: info_dict}`` mapping built from every
    registered media type's :attr:`~MediaType.demo_datasets` list.

    Each value is a dict with the keys expected by the datasets route:
    ``label``, ``description``, ``categories``, ``media_type``, and
    optionally ``source``.
    """
    result: dict = {}
    for mt in _registry.values():
        for ds in mt.demo_datasets:
            entry: dict = {
                "label": ds.label,
                "description": ds.description,
                "categories": ds.categories,
                "media_type": mt.type_id,
            }
            if ds.source:
                entry["source"] = ds.source
            result[ds.id] = entry
    return result


# ------------------------------------------------------------------
# Register all built-in media types
# ------------------------------------------------------------------
# To add a new media type, import its class here and call register().
# The four imports below are the complete list of built-in types.

from vistatotes.media.audio.media_type import AudioMediaType  # noqa: E402
from vistatotes.media.image.media_type import ImageMediaType  # noqa: E402
from vistatotes.media.text.media_type import TextMediaType    # noqa: E402
from vistatotes.media.video.media_type import VideoMediaType  # noqa: E402

register(AudioMediaType())
register(VideoMediaType())
register(ImageMediaType())
register(TextMediaType())

__all__ = [
    "MediaType",
    "DemoDataset",
    "register",
    "get",
    "get_by_folder_name",
    "all_types",
    "all_demo_datasets",
]
