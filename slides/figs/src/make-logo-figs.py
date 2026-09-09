#!/usr/bin/env python
"""One slide: eight results for *Coke logo*, arriving one at a time.

    python slides/figs/src/make-logo-figs.py

Writes `figs/logo-grid.webp` and its six build stages — the slide is a build, so
the room is asked about each result as it lands rather than being handed all
eight at once. The whole argument lives in the presenter notes
(`fragments/logo-grid.md`): *are these the same? …and this one? …what if the
colours invert? …this can't count, right?* Every answer is defensible and no
two people give the same set of them, which is the point.

There is nothing to compute here. The figure is a **compositor**: it lays the
committed thumbnails in `logo-src/` onto a fixed 3x3 grid and saves one stage
per reveal, with later cells simply empty. That fixed grid is the whole reason
this is a generated figure rather than eight `<img>` tags in the fragment — a
build marker reveals by *truncating* the fragment, so an HTML grid would
reflow on every page and the images would shuffle around the slide instead of
arriving in place. `slides/STYLE.md` is explicit that a reveal adds ink and
does nothing else.

The top-left cell is left empty for the slide's headline, which is what buys
this figure a title: eight tiles in a 4x2 grid would fill the corner
`slide_figure.TITLE_NOTCH_PX` reserves, and a full-bleed figure that cannot
spare that corner has to carry no title at all.
"""

from __future__ import annotations

import functools
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import (  # noqa: E402
    FULL_BLEED,
    save,
)

OUT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent / "logo-src"

INK = "#14181f"
SOFT = "#5b6472"
RULE = "#d8dee6"

#: WebP, not PNG, and this is the one figure in the deck that has to be.
#: Every other generated figure here is line art on white, which is what PNG is
#: for; these tiles are eight JPEG *photographs* — gradients, drop shadows and
#: compression noise, upscaled — which is the case `slides/README.md` reserves
#: WebP for, and the seven cumulative stages each carry every earlier tile
#: again. As PNG the group weighs 5.0 MB against a stated budget of about
#: 150 KB per figure. Marp rasterises through Chromium, which reads WebP
#: natively, and the deck already ships its two UI screenshots this way.
FIGURE_FORMAT = "webp"

#: Points per drawing unit, shared with the calibration schematics so type set
#: here is the size it is there. See `make-calib-figs.FLOW_UNIT_PT`.
UNIT_PT = 38.0

#: Exactly 16:9, and every stage is written with `tight=False`, so the canvas
#: maps one-to-one onto the 1280x720 slide and the grid geometry below is in
#: slide pixels divided by 64.65.
CANVAS = (19.8, 11.0)

COLS, ROWS = 3, 3

#: How much of the canvas's bottom edge the grid keeps clear. The slide draws
#: its own page number in the bottom-right corner, and the last row of tiles
#: otherwise runs underneath it.
GRID_FOOT = 0.75

CELL_W, CELL_H = CANVAS[0] / COLS, (CANVAS[1] - GRID_FOOT) / ROWS

#: How much of each cell is margin rather than image. Wide enough that two
#: tiles never touch — several of these thumbnails are white-on-white at the
#: edges and would otherwise read as one wide picture.
CELL_PAD = 0.42

#: The eight results, in the order the slide reveals them, as
#: `(file, what it is, size class)`. The file is looked up in `logo-src/`; a slot whose
#: file is not there yet draws a dashed placeholder carrying its description,
#: so the deck builds and the layout can be reviewed before every asset has
#: arrived. **A placeholder is not a figure** — re-run this script once the
#: file lands and commit the result.
#:
#: The order is the argument's, not the search engine's, and it is also the
#: **grid's reading order**: the two badges, then the three red-field marks,
#: then the three that each break a *different* attribute — the colour, the
#: typeface, the product. Reveal order and layout order are the same tuple on
#: purpose, so a slide that fills left-to-right, top-to-bottom cannot disagree
#: with the order the presenter narrates. See the fragment's notes.
#:
#: The **size class** is what stops the grid from making a point it does not
#: mean. These eight arrived at eight unrelated resolutions and crops, and
#: fitted individually to their cells they came out at eight different sizes —
#: so the black script printed a third larger than the red one it is supposed
#: to be identical to, and the room reads a size difference as a claim. Members
#: of a class are drawn at one common width (see `_class_widths`); the four
#: classes are the four parallel pairs the slide actually argues over.
RESULTS = (
    ("02-red-disc.jpg", "the round red badge", "disc"),
    ("03-disc-with-bottle.jpg", "that badge, with a contour bottle", "disc"),
    ("01-wordmark-on-red.jpg", "wordmark, white on a solid red field", "panel"),
    ("04-wordmark-ribbon.jpg", "wordmark over the dynamic ribbon", "panel"),
    ("05-script-red-on-white.jpg", "the script, red on white", "script"),
    ("06-script-black.jpg", "the script, in flat black", "script"),
    ("07-coke-sans.png", "“Coke”, in a heavy sans", "coke"),
    ("08-diet-coke.jpg", "Diet Coke", "coke"),
)

#: How wide a placeholder's description may run before it wraps, in
#: characters. Measured against the cell rather than guessed: a placeholder
#: that overflows its own box is the one thing it must not do, since its whole
#: job is to show what the finished layout will look like.
PLACEHOLDER_CHARS = 24

#: Which grid cell each result lands in, as `(col, row)` with row 0 at the top,
#: in `RESULTS` order. The top-left cell is skipped: that is where the headline
#: goes, which is why the top row holds two results and the others hold three.
CELLS = ((1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2))

#: Rows whose tiles are aligned on a common **baseline** rather than centred.
#: The bottom row is “Coca-Cola”, “Coke” and “Diet Coke”, and centring them
#: lined up the wrong thing: *Diet* rides above the cap line, so centring
#: pushed 08's “Coke” down until it sat lower than 07's. Sharing a baseline
#: lines up the word the three tiles have in common. The row's tallest tile
#: still centres in its cell and the others hang from its baseline, so the row
#: does not sink to the bottom of the grid.
BASELINE_ROWS = frozenset({2})

#: What fraction of a tile's width is the logo's *lettering* rather than the
#: field it is printed on. A bare wordmark is cropped to its own letters, so it
#: is 1.0 by construction; a red panel carries margin around the script, so
#: drawing panel and wordmark at one width prints the panel's script visibly
#: smaller than the same script standing alone — which is a size difference
#: the slide does not mean, on a slide about whether two logos are the same.
#:
#: 0.86 is measured, not chosen: the white script spans 0.828 of the trimmed
#: width of `01` and 0.894 of `04`, and this is their mean. To re-take it, mask
#: the near-white pixels of a trimmed panel and read the span of the columns
#: holding a real number of them — for `04` restrict to the top 62% of the
#: height first, or the dynamic ribbon (which runs the panel's full width, and
#: is part of the mark rather than margin) reads as lettering and returns 1.0.
LETTERFORM = {"panel": 0.86, "disc": 1.0, "script": 1.0, "coke": 1.0}

#: The classes whose *lettering* is normalised against one another, rather than
#: their tiles. These are the marks that are all fundamentally a word, so the
#: word is what should match. Discs are deliberately out: a badge is a badge,
#: and shrinking one until its script matched a bare wordmark's would leave a
#: tiny disc floating in its cell.
MATCHED_CLASSES = frozenset({"panel", "script", "coke"})

#: How many pages the slide is. The first shows **two** results, because the
#: opening question is a comparison — "are these the same?" needs two things to
#: be the same as each other — and every later page adds one.
STAGES = len(RESULTS) - 1

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 15,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def _cell_box(index: int) -> tuple[float, float, float, float]:
    """Result *index*'s drawable rectangle as `(x0, y0, x1, y1)`, padded."""
    col, row = CELLS[index]
    x0 = col * CELL_W + CELL_PAD
    x1 = (col + 1) * CELL_W - CELL_PAD
    y1 = CANVAS[1] - row * CELL_H - CELL_PAD
    y0 = CANVAS[1] - (row + 1) * CELL_H + CELL_PAD
    return x0, y0, x1, y1


#: How far a pixel may sit from the corner colour and still count as border.
#: These are JPEG thumbnails, so a "white" margin is white plus ringing, and an
#: exact-match trim finds nothing at all on half of them.
TRIM_TOLERANCE = 12


def _flattened(image: Image.Image) -> Image.Image:
    """`image` as RGB, with any transparency composited onto white.

    A bare `.convert("RGB")` resolves a transparent pixel to whatever its
    palette index happens to hold, which on one of these sources is black: the
    logo arrived as a palette PNG with a transparent ground, and converting it
    directly produced a black rectangle that then defeated the border trim and
    would have printed a black tile onto a white slide. White is the right
    ground because the slide is white.
    """
    if image.mode not in ("RGBA", "LA", "P"):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    ground = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(ground, rgba).convert("RGB")


def _trimmed(image: Image.Image) -> Image.Image:
    """`image` with its uniform border cropped off.

    The sources are search-result thumbnails, which are padded to a squarish
    box: several are a wordmark occupying a third of their own height with
    white above and below. Fitted untrimmed, that padding is what touches the
    cell and the art is drawn at a third of the size the slide is paying for.
    Trimming is what makes the eight tiles comparable to *each other*, too —
    otherwise a tile's apparent size records how much whitespace its thumbnail
    happened to carry.

    Returns the image unchanged when the trim finds nothing (art that already
    bleeds to all four edges) or everything (a solid tile, which cannot
    happen here but would otherwise crop to nothing).
    """
    border = Image.new("RGB", image.size, image.getpixel((0, 0)))
    difference = ImageChops.difference(image, border).convert("L")
    box = difference.point(lambda v: 255 if v > TRIM_TOLERANCE else 0).getbbox()
    return image if box is None else image.crop(box)


@functools.lru_cache(maxsize=None)
def _art_size(index: int) -> tuple[int, int] | None:
    """Result *index*'s trimmed art size in pixels, or None if its file is absent."""
    path = SRC / RESULTS[index][0]
    if not path.exists():
        return None
    with Image.open(path) as image:
        return _trimmed(_flattened(image)).size


@functools.lru_cache(maxsize=None)
def _row_baseline_height(row: int) -> float:
    """The drawn height of the tallest tile in a baseline-aligned *row*."""
    heights = []
    for index, (col_row, (_, _, size_class)) in enumerate(zip(CELLS, RESULTS, strict=True)):
        art = _art_size(index)
        if col_row[1] != row or art is None:
            continue
        width, height = art
        heights.append(_class_widths()[size_class] * height / width)
    return max(heights, default=0.0)


def _tile_bottom(index: int, y0: float, y1: float, drawn_h: float) -> float:
    """Where result *index*'s art sits vertically in its cell.

    Centred, except in a `BASELINE_ROWS` row, where every tile hangs from the
    baseline of the row's tallest one — so the word three tiles share lines up
    rather than their bounding boxes.
    """
    row = CELLS[index][1]
    if row not in BASELINE_ROWS:
        return (y0 + y1) / 2 - drawn_h / 2
    return (y0 + y1) / 2 - _row_baseline_height(row) / 2


@functools.lru_cache(maxsize=None)
def _class_widths() -> dict[str, float]:
    """The drawn width, in canvas units, shared by every member of each size class.

    One width per class rather than one per image, because "the same size" is
    the whole point: 05 and 06 are the *same artwork* in two colours, and 07
    and 08 differ only by a script word riding above the cap line. Fitted
    independently they came out at different sizes, and on a slide asking the
    room whether two logos are the same thing, a gratuitous size difference is
    an answer nobody meant to give.

    **Width** is the shared dimension, not height and not area, and 07/08 is
    why. "Diet Coke" is taller than "Coke" only because *Diet* rides above the
    cap line, so equalising height (or area, which follows height here) would
    shrink the word "Coke" in 08 relative to 07 — the one thing the pair has in
    common, drawn at two sizes. Equalising width leaves them matched. For the
    other three classes the members share an aspect closely enough that all
    three rules agree.

    The width is the largest one every member can *fit*: an image is capped by
    the cell's width, and by the cell's height once its own aspect is applied,
    so the class is bound by whichever member runs out of room first.
    """
    cell_w = CELL_W - 2 * CELL_PAD
    cell_h = CELL_H - 2 * CELL_PAD
    widths: dict[str, float] = {}
    for index, (_, _, size_class) in enumerate(RESULTS):
        art = _art_size(index)
        if art is None:
            continue
        width, height = art
        fits = min(cell_w, cell_h * width / height)
        widths[size_class] = min(widths.get(size_class, fits), fits)

    # Then equalise the *lettering* across `MATCHED_CLASSES`, not the tiles: a
    # class that carries margin around its script needs a wider tile to print
    # the same size word. The shared lettering width is the largest every
    # matched class can still fit, so nothing overflows its cell.
    matched = [c for c in widths if c in MATCHED_CLASSES]
    if matched:
        lettering = min(widths[c] * LETTERFORM[c] for c in matched)
        for size_class in matched:
            widths[size_class] = lettering / LETTERFORM[size_class]
    return widths


def _draw(ax: plt.Axes, index: int) -> None:
    """Draw result *index* in its cell at its class's width, centred, or a placeholder."""
    x0, y0, x1, y1 = _cell_box(index)
    name, described, _ = RESULTS[index]
    path = SRC / name
    if not path.exists():
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor=SOFT,
                linewidth=1.4,
                linestyle=(0, (4, 4)),
            )
        )
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            "awaiting\n" + "\n".join(textwrap.wrap(described, PLACEHOLDER_CHARS)),
            ha="center",
            va="center",
            fontsize=15,
            color=SOFT,
            linespacing=1.3,
        )
        return

    with Image.open(path) as image:
        pixels = _trimmed(_flattened(image))
        width, height = pixels.size
        # Not "fit to the cell" — every member of a size class is drawn at the
        # one width that class agreed on, so two tiles the slide calls parallel
        # are the same size however their sources happened to be cropped.
        # Aspect is preserved, so the height follows.
        drawn_w = _class_widths()[RESULTS[index][2]]
        drawn_h = drawn_w * height / width
        cx = (x0 + x1) / 2
        bottom = _tile_bottom(index, y0, y1, drawn_h)
        ax.imshow(
            pixels,
            extent=(cx - drawn_w / 2, cx + drawn_w / 2, bottom, bottom + drawn_h),
            aspect="auto",
            interpolation="lanczos",
            zorder=2,
        )


def _stage(stage: int) -> plt.Figure:
    """The first *stage* reveals (1-based, cumulative): two results, then one more each."""
    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.set_axis_off()
    for index in range(stage + 1):
        _draw(ax, index)
    return fig


def main() -> None:
    for stage in range(1, STAGES):
        save(_stage(stage), OUT, f"logo-grid.build{stage}.{FIGURE_FORMAT}", column=FULL_BLEED, tight=False)
    save(_stage(STAGES), OUT, f"logo-grid.{FIGURE_FORMAT}", column=FULL_BLEED, tight=False)


if __name__ == "__main__":
    main()
