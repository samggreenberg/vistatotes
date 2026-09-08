#!/usr/bin/env python
"""COCO val2017 on demand, and the two photo corpora the deck's figures need.

    python slides/figs/src/coco_fixture.py photos          # -> prints the path
    python slides/figs/src/coco_fixture.py photo-regions

The deck's running example is **book** (see `make-book-figs.py` for why), so
the screenshots have to be a real session hunting books, and that needs a
corpus where books are a real concept with real near-misses in the pile.
Caltech-101, which these shots used to be taken against, has no such category;
COCO does, and it is the set whose `book` annotation the `vg_scale` review
found hardest to agree with (`docs/experiments/2026-08-25-vg-scale/DATASHEET.md`).

So: download COCO val2017 into `data/coco-val2017/` once, and materialise
fixture folders under `data/slide-fixtures/` filed by category, exactly the
shape `shoot-ui-figs.mjs` and the app's `server_folder` importer expect. Both
steps are idempotent — a re-run after a GUI change re-uses what is on disk —
and neither writes anything into the repo.

Selection is a pure function of the download: every category takes the images
whose largest box of that class covers most of the frame, ties broken by COCO
image id, so a rebuild produces the same corpus rather than reshuffling it.
Frames keep their COCO file names, because that is what makes a chosen frame
nameable: `shoot-ui-figs.mjs` picks the photo it votes on and the photo it puts
in the centre viewer by id, and box area is not a good enough proxy for "this
photograph is about a book" to pick them automatically — the frame with the
largest `book` box in val2017 is a game manual in a Wii box.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

CORPUS = _REPO_ROOT / "data" / "coco-val2017"
FIXTURES = _REPO_ROOT / "data" / "slide-fixtures"
IMAGES = CORPUS / "val2017"
ANNOTATIONS = CORPUS / "annotations" / "instances_val2017.json"

# Plain HTTP: the COCO host's certificate does not match its own name through a
# TLS-terminating proxy, and nothing downstream trusts these bytes with
# anything — they are public research photographs that end up as pixels in a
# committed figure.
IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

#: The two corpora, as `{category: how many images}`.
#:
#: `photos` is the three-panel shot's pile: books, rare-ish, among a few
#: hundred photographs of other things. The negatives are deliberately weighted
#: toward the rectangular, printed and shelved — a laptop, a monitor, a
#: keyboard, a phone, a clock — because a corpus whose only non-books are
#: giraffes makes the detector look better than it is, and the deck's whole
#: point is that the boundary is hard.
#:
#: `photos-prod` is the *production* pile the Find slide scores: the same fifteen
#: subjects in the same proportions, and not one frame in common with `photos`.
#: Sharing nothing is the slide's whole claim — the detector was voted on one
#: pile and is now ranking media it has never seen — so it is enforced here
#: rather than left to a comment (see `DISJOINT_FROM`).
#:
#: `photo-regions` is much smaller: it is embedded with DINOv2 patch, which is
#: many times the work per image, and the shot only ever shows one item and the
#: strip of thumbnails beside it.
PLANS = {
    "photos": {
        "book": 44,
        "laptop": 18,
        "tv": 16,
        "keyboard": 14,
        "cell phone": 14,
        "clock": 14,
        "vase": 12,
        "chair": 12,
        "pizza": 12,
        "dog": 12,
        "bird": 12,
        "bicycle": 12,
        "bus": 12,
        "umbrella": 12,
        "sandwich": 12,
    },
    "photos-prod": {
        "book": 40,
        "laptop": 16,
        "tv": 16,
        "keyboard": 12,
        "cell phone": 16,
        "clock": 16,
        "vase": 12,
        "chair": 16,
        "pizza": 14,
        "dog": 14,
        "bird": 14,
        "bicycle": 14,
        "bus": 14,
        "umbrella": 14,
        "sandwich": 12,
    },
    "photo-regions": {
        "book": 14,
        "laptop": 8,
        "tv": 6,
        "dog": 6,
        "bicycle": 6,
    },
}


#: Corpora that must not share a single frame with another, as
#: `{plan: the plan it stands clear of}`.
#:
#: The Find slide claims the detector is ranking media nobody voted on. A
#: `photos-prod` built by re-running the same "largest box first" selection
#: would hand back exactly the frames `photos` already took, and the claim
#: would be false in the one figure that makes it. So the reserved plan's
#: frames are struck out of the candidate list before the second plan chooses,
#: which leaves `photos-prod` the next-best frames of each category rather than
#: a reshuffle.
DISJOINT_FROM = {"photos-prod": "photos"}


def ensure_corpus() -> Path:
    """Download COCO val2017 into `data/coco-val2017/` if it is not there yet."""
    if IMAGES.is_dir() and any(IMAGES.iterdir()) and ANNOTATIONS.exists():
        return CORPUS
    CORPUS.mkdir(parents=True, exist_ok=True)
    for url, member in ((ANNOTATIONS_URL, "annotations/instances_val2017.json"), (IMAGES_URL, None)):
        archive = CORPUS / url.rsplit("/", 1)[-1]
        if not archive.exists():
            print(f"downloading {url} ...", file=sys.stderr)
            with urllib.request.urlopen(url) as response, archive.open("wb") as out:  # noqa: S310
                while chunk := response.read(1 << 20):
                    out.write(chunk)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(CORPUS, members=[member] if member else None)
        archive.unlink()
    return CORPUS


def _roster(plan: dict[str, int], reserved: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """Which COCO frames fill each category of *plan*, deterministically.

    A frame lands in the category whose box covers most of it, and only in one:
    a photograph of a laptop beside a stack of books is a book photograph as
    far as this corpus is concerned, and filing it under `laptop` would put a
    true positive in the negatives. `book` is resolved first for that reason.

    *reserved* is a set of COCO file names another plan has already claimed;
    they are struck out before this plan chooses, which is how a disjoint
    corpus is built (see `DISJOINT_FROM`).
    """
    data = json.loads(ANNOTATIONS.read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    frames = {i["id"]: i for i in data["images"]}
    # Largest box of each class in each image, as a fraction of the frame.
    share: dict[int, dict[str, float]] = {}
    for ann in data["annotations"]:
        frame = frames[ann["image_id"]]
        area = frame["width"] * frame["height"]
        cell = share.setdefault(ann["image_id"], {})
        name = names[ann["category_id"]]
        cell[name] = max(cell.get(name, 0.0), ann["area"] / area)

    taken: set[int] = set()
    roster: dict[str, list[str]] = {}
    for category, count in plan.items():
        candidates = [
            (-cell[category], image_id)
            for image_id, cell in share.items()
            if category in cell
            and image_id not in taken
            and frames[image_id]["file_name"] not in reserved
            and (category == "book" or "book" not in cell)
        ]
        candidates.sort()
        chosen = [image_id for _, image_id in candidates[:count]]
        if len(chosen) < count:
            raise SystemExit(f"COCO val2017 has only {len(chosen)} usable {category!r} frames, wanted {count}")
        taken.update(chosen)
        roster[category] = [frames[image_id]["file_name"] for image_id in chosen]
    return roster


def _reserved_for(name: str) -> frozenset[str]:
    """The frames *name* must not take, because another corpus holds them."""
    other = DISJOINT_FROM.get(name)
    if other is None:
        return frozenset()
    return frozenset(f for files in _roster(PLANS[other]).values() for f in files)


def build_fixture(name: str) -> Path:
    """Materialise one fixture under `data/slide-fixtures/`, and return its path."""
    root = FIXTURES / name
    if root.is_dir() and any(root.iterdir()):
        return root
    ensure_corpus()
    for category, files in _roster(PLANS[name], _reserved_for(name)).items():
        folder = root / category.replace(" ", "-")
        folder.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy(IMAGES / file, folder / file)
    print(f"built corpus {name}", file=sys.stderr)
    return root


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in PLANS:
        raise SystemExit(f"usage: coco_fixture.py {{{'|'.join(PLANS)}}}")
    print(build_fixture(sys.argv[1]))
