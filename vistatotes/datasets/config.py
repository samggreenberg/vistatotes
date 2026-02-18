"""Dataset configurations — built from the media type registry.

``DEMO_DATASETS`` is assembled at import time from every registered
:class:`~vistatotes.media.base.MediaType`'s
:attr:`~vistatotes.media.base.MediaType.demo_datasets` list.  Adding a new
media type (and registering it in ``vistatotes/media/__init__.py``)
automatically makes its demo datasets appear here with no further edits.
"""

from vistatotes.media import all_demo_datasets

DEMO_DATASETS: dict = all_demo_datasets()
