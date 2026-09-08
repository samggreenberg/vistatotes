#!/usr/bin/env python
"""Introduction figures for the VTSearch decks.

Run from the repo root:

    python slides/figs/src/make-intro-figs.py

One figure lives here: `vote-boundary`, the picture of what a threshold
actually *does*. The deck's other mechanism figures are drawn in score space —
a number line with a cut on it — which is the right space for a talk about
where the cut goes, and the wrong one for a talk about what the cut is *for*,
because it cannot show why the item the user is asked about next is the one it
is.

So this figure works in *item* space: every media item is a point, and the
detector is the closed curve around the ones it currently calls a match. That
is a genuine 2D analogue rather than a picture of the shipped model — VTSearch
trains a linear SVM in embedding space, where the boundary is a hyperplane, and
a hyperplane in two dimensions is a straight line that cannot enclose anything.
An RBF SVM on two dimensions is the same object with the curvature the audience
would otherwise have to imagine, so that is what is fitted here: the boundary on
every stage is a real `sklearn` decision contour over the votes shown, the
looser and tighter cuts are real level sets of that same decision function, and
the item selected next is really the unlabeled point nearest the boundary — the
app's own `Hard` rule.

**With one exception, and it is deliberate.** The retrained boundary is the two
real fits *spliced*: the new one where the answered vote reaches, the old one
everywhere else, crossfaded between (`_Blended`). A plain refit moves the far
side of the loop as well, by an amount that is the solver rebalancing its
intercept rather than anything the vote means, and a page whose whole job is
"watch the curve reach out and take that item in" cannot afford a second thing
moving at the same time (#3763).

**The slide carries the whole argument, so the figure carries seven stages.**
The pile; the votes and the detector they imply; the same detector cut looser
and tighter, which is the threshold's first job — what comes back; the two
items nobody should be asked about; the one item that is worth a question,
which is the threshold's second job; the answer, and the boundary redrawn; and
the next question, which exists only because the boundary moved. Slides 7 and 8
of the deck used to make the "one line, two jobs" point in the abstract, on a
schematic and then on a sentence. Made here, on the field, it costs no slides
at all (#3246).

**Where the items sit is the one thing the figure chooses.** Which items are
matches, and where the curve goes, follow from the data and the fit; the 2D
coordinates are arbitrary, so `_scene` spends them on making the drawing say
what is true. It settles the field until no boundary passes *through* an item —
a curve crossing a circle draws an item the detector cut in half, where the
claim is that the item sits near the line — and it places the item the app asks
about just outside the curve it is nearest, so that "this is the one it cannot
call" is readable and not merely true. Outside matters twice over: an item that
is already inside the boundary is one the model already calls a match, so
answering it Good would teach it nothing and the retrained curve would not
move. Assertions cover every claim that survives the settling.

**The field fills the slide, and leaves one hole.** The point of the opening
stage is a corpus too big to look through, so a drawing that uses the middle
half of a 16:9 slide is arguing against itself. The field is therefore laid out
across the whole slot — except the top-left corner, which every full-bleed
slide reserves for its headline (`slide_figure.TITLE_NOTCH_PX`). Items are
rejected from that rectangle rather than the title being moved, because the
notch is a fixed standard and a headline that dodges each figure stops being a
headline. The figure is saved untrimmed (`tight=False`) so the PNG *is* the
16:9 window and the reserve lands where the arithmetic says it does.

**On the build rule.** `slides/STYLE.md` says a reveal adds ink and nothing
moves or restyles between pages. This figure breaks that in exactly three
places, all because the restyle *is* the mechanism: the item under
consideration grows a question mark and then becomes a check; the loose and
tight cuts are scaffolding for one beat and leave once the argument has moved
on from what comes back to what gets asked; and the boundary moves once the
retrain has happened (the previous boundary stays on the slide, faded, so what
the audience sees is where it went, not a curve teleporting). Everything else —
the window, the crop, every item's position — is pinned across all seven pages.
"""

import functools
import sys
from pathlib import Path

# Ensure the repo root (where app.py lives) is importable no matter the cwd:
# ``python slides/figs/src/x.py`` only puts the script's own dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, notch_box, save  # noqa: E402

OUT = Path(__file__).resolve().parent.parent

INK = "#14181f"
SOFT = "#5b6472"
BLUE = "#0b5fa5"  # the detector's boundary — the shipped decision
BAND = "#dce7f2"  # the strip between the loose and the tight cut
RED = "#b91c1c"  # the Bad side
GREEN = "#0d8a5f"  # the Good side
GHOST = "#aab3c0"  # the boundary as it was before the retrain

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 17,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)

#: The figure is a 16:9 canvas — the whole full-bleed slot — drawn at this many
#: printed points per unit, which fixes the size everything else is expressed
#: in. Every stage is saved untrimmed, so the PNG has exactly this aspect and
#: the title notch falls exactly where `slide_figure` says it does.
UNIT_PT = 46.0
CANVAS = (16.0, 9.0)

#: Item glyph radius, in figure units. Checks and crosses are drawn to the same
#: half-width, so a voted item occupies the same patch of the field as the
#: hollow circle it replaces and the field's texture does not change density
#: as votes accumulate.
R = 0.17

#: How many items the field holds, and how far apart they are laid out. A
#: corpus the user could look through by hand is not the situation this deck
#: opens on, so the field is dense enough to read as "too many" from the back
#: of the room and no denser: below `ITEM_APART` two circles start to read as
#: one smudge.
FIELD_ITEMS = 96

#: How far inside the boundary the unlabeled item the figure calls *obvious*
#: has to sit, in figure units — well over two glyph radii, so the gap is
#: visible as a gap at slide size rather than inferred from the maths.
OBVIOUS_GAP = 0.62

#: Where the item the app is asking about sits, relative to the boundary, in
#: figure units. A curve that passes *through* a circle draws an item the
#: detector has cut in half; what the slide claims is that the item sits
#: **near** the line, which is a different picture and one the audience reads
#: off the gap. Wide enough to be a gap at slide size, narrow enough that "that
#: one is right on the line" is still the obvious reading.
CURVE_CLEAR = R + 0.07

#: The floor for an item caught in the corridor between the two boundaries, in
#: figure units — half that corridor is all such an item can have, and asking
#: for more is asking for a layout that does not exist. Above `CURVE_CLEAR`, so
#: the item the app asks about is still the nearest thing to the line, and well
#: above `R`, so no boundary is ever drawn through a circle: an item a curve
#: crosses is an item the detector cut in half, which is a picture of something
#: that does not happen.
CURVE_TOUCH = R + 0.10

#: How much room every *other* item gets, in figure units. Strictly more than
#: `CURVE_CLEAR`, because "this is the one it cannot call" is a claim about the
#: item being nearest the line, and a field where everything sits at the same
#: distance makes that claim unreadable however true it is.
CURVE_ROOM = R + 0.26

#: How much room every other item owes the *retrained* boundary, in figure
#: units. Less than `CURVE_ROOM`, and deliberately: the two boundaries run
#: close together over most of their length, so asking a dense field for the
#: full room from both carves one corridor wide enough to jam items inside it.
#: The retrained curve only has to avoid drawing through a circle — which item
#: sits nearest *it* is said by the question mark, not by the spacing.
CURVE_ROOM_AFTER = R + 0.14

#: Minimum centre-to-centre distance between any two items, in figure units.
#: The field is drawn with more room than this; it is here because pushing
#: items off a curve can push two of them together.
ITEM_APART = 0.56

#: How far past the minimum a nudge lands, in figure units. Settling *onto* the
#: limit never terminates: the curve is sampled, so the measured distance
#: wobbles either side of the target and every pass finds the same item a
#: hair short of it again.
OVERSHOOT = 0.03

#: The movement below which the field counts as settled, in figure units —
#: about two rendered pixels in the deck, so nothing that survives it is
#: visible. A pinned item never stops moving entirely: pinning it moves the fit
#: it is a training point for, which moves the curve it is pinned to.
SETTLED = 0.006

#: How much of a pin's correction is applied per pass. Applying all of it makes
#: that same feedback loop ring instead of converging.
PIN_DAMPING = 0.55

#: How much of an ordinary shove is applied per pass, for the same reason at a
#: different scale: in a field this dense an item is usually crowded by a curve
#: and two neighbours at once, and moving it the whole way clear of one puts it
#: inside another. Applied undamped the three rules take turns and the field
#: oscillates forever a handful of violations short of settled; damped, they
#: share the correction and it converges.
SHOVE_DAMPING = 0.5

#: Spacing of the sampled points that stand in for a drawn boundary, in figure
#: units — well under `SETTLED`, so sampling is never what a distance turns on.
CONTOUR_STEP = 0.02

#: The kernel width of the stand-in detector, in figure units. Set by the
#: length scale the drawing should have rather than by the data: a small
#: `gamma` bends the curve around every vote and produces the lumpy,
#: reach-around shape #3246 objected to, where the reader is asked to believe a
#: dent in the boundary that nothing in the field explains. At this width the
#: curve is a smooth closed loop that separates the checks from the crosses
#: with room on both sides, which is what the slide is claiming.
#:
#: **It also decides how much of the retrain the audience has to ignore**, and
#: that is what set the number (#3763). An RBF SVM's decision function is a sum
#: of kernels plus a bias, and the bias is refitted along with everything else:
#: adding one Good vote pushes the curve out around that vote *and* slides the
#: whole level set, so the far side of the loop creeps inward for reasons
#: nothing on the slide explains. The build asks the room to watch one thing —
#: the curve reaching out to cover the item they just answered — and every unit
#: of far-side creep is a second thing moving at the same time.
#:
#: A wider kernel spreads one vote's influence further and makes that creep
#: worse; a narrower one localises it, at the cost of the lumpiness above.
#: Measured on the settled field, as (expansion within 1.5 units of the answered
#: item) against (motion more than 4 units away):
#:
#:     width   expansion   far motion   far max
#:      1.10       0.62        0.14       0.15
#:      1.15       0.63        0.11       0.14
#:      1.25       0.65        0.05       0.09
#:      1.30       0.65        0.05       0.12
#:      1.35       0.66        0.07       0.19
#:      1.45       0.66        0.15       0.27
#:
#: 1.30 sits in the floor of that trough: the intended motion is what it always
#: was and the distraction is a third of what 1.45 paid. Below it the far field
#: creeps again *and* the retrained loop loses convexity — it encloses 0.984 of
#: its own hull's area at 1.30 and 0.962 at 1.10 — which is the shape complaint
#: above coming back. Re-measure both columns if the field or the votes change.
KERNEL_SCALE = 1.30

#: How far the looser and the tighter cut sit from the shipped one, in figure
#: units. Set geometrically rather than as a fraction of the votes' margin,
#: because what the stage has to do is *look* like three cuts of one detector:
#: too close and the strip is a fat line, too far and the loose cut stops being
#: a closed curve at all and runs off the slide. The level that realises it is
#: solved for, and capped so that neither cut can misread a vote — the whole
#: point of the pair is that the calibration data cannot choose between the
#: three, and the choice is the user's.
BAND_WIDTH = 0.62

#: The most of the votes' margin a band cut may spend, as a fraction of the
#: smallest margin any vote has, and of how far the decision function falls
#: away from every training point. The first keeps the checks and crosses on
#: the sides they were voted onto; the second keeps the loose cut a closed
#: curve rather than a level the far field also clears.
BAND_MARGIN_CAP = 0.85

#: This figure's own title reserve, in slide pixels — the deck standard's
#: rectangle with its *height* trimmed to the headline this slide actually
#: carries. "Rock the Vote" is one line and its box measures 56.8px, so
#: `slide_figure.TITLE_NOTCH_PX`'s 200px reserve — sized for the deck's
#: longest, four-line headline — left 100px of band under the title with no
#: title in it and, because the field is rejected from the whole reserve, no
#: items either. On a schematic that costs nothing; here the field *is* the
#: slide, and a hole in it reads as a mistake rather than as a margin (#3254).
#: 88px is the measured box plus one `OBJECT_GAP_PT` (16pt renders at ~28px on
#: this figure), so the nearest circle still clears the headline by a gap of
#: the deck's own standard size. Re-measure if the headline changes; the
#: recipe is in `slides/STYLE.md`.
VOTE_NOTCH_PX = (60.0, 42.0, 300.0, 88.0)

#: The nine pages of the build. See `vote_boundary_fig`.
VOTE_BOUNDARY_STAGES = 9

#: The page that goes *back*: the flashback re-draws stage 5's picture — the
#: first boundary, the item it cannot call — with the loose and tight cuts
#: added, to say that the threshold was already choosing the question that got
#: us here. Held as a constant because two functions have to agree on which
#: page is not simply "the first N steps".
FLASHBACK_STAGE = 9
#: The step the flashback re-draws.
FLASHBACK_OF = 5


# ──────────────────────────────────────────────────────────────────────────────
# The field of items
# ──────────────────────────────────────────────────────────────────────────────


def _notch_rect() -> tuple[float, float, float, float]:
    """The slide's title reserve, in the figure's own drawing units.

    `slide_figure.notch_box` returns figure fractions, which are exact here
    because the figure is saved untrimmed at the slot's own 16:9 aspect: the
    PNG *is* the window, so a fraction of the image is the same fraction of
    the canvas.
    """
    box = notch_box(*CANVAS, FULL_BLEED, VOTE_NOTCH_PX)
    assert box is not None, "a 16:9 full-bleed figure always overlaps the notch"
    x0, y0, x1, y1 = box
    width, height = CANVAS
    return x0 * width, y0 * height, x1 * width, y1 * height


def _in_notch(p: np.ndarray, margin: float) -> bool:
    x0, y0, x1, y1 = _notch_rect()
    return x0 - margin < p[0] < x1 + margin and y0 - margin < p[1] < y1 + margin


#: How many candidate positions each item is chosen from. This is Mitchell's
#: best-candidate rule: draw this many uniform points, keep whichever is
#: furthest from everything already placed. The dart-throwing sampler it
#: replaced accepted the *first* candidate that cleared a minimum separation,
#: which is a weaker thing to ask — it forbids clumps but does nothing about
#: voids, and on a field this sparse (96 discs on 144 square units is 27%
#: packing) it left holes several items wide that read as regions the drawing
#: was making a claim about (#3301). Best-candidate spends its randomness on
#: the emptiest place instead, so the field stays irregular — it must not read
#: as a lattice — without opening a gap the eye stops on.
FIELD_CANDIDATES = 48


@functools.lru_cache(maxsize=1)
def _blue_noise() -> np.ndarray:
    """A fixed field of items in 2D, spread evenly without being regular.

    Blue-noise rather than uniform: a uniform draw clumps *and* voids, and both
    read as structure at slide size — a clump as one blurred object, a void as
    somewhere the figure means something by. Laid out across the whole 16:9
    canvas apart from the title reserve, because the opening stage's claim is
    that there is too much of this to look through.
    """
    rng = np.random.default_rng(11)
    margin = 0.55
    low = np.array([margin, margin])
    high = np.array(CANVAS) - margin
    pts: list[np.ndarray] = []
    for _ in range(FIELD_ITEMS):
        best: np.ndarray | None = None
        best_gap = -1.0
        for _ in range(FIELD_CANDIDATES):
            p = rng.uniform(low, high)
            if _in_notch(p, R + 0.22):
                continue
            gap = min((float(np.hypot(*(p - q))) for q in pts), default=float("inf"))
            if gap > best_gap:
                best, best_gap = p, gap
        if best is None:
            raise SystemExit("every candidate landed in the title reserve — the margins are wrong")
        pts.append(best)
    return np.array(pts)


def _surrounded(p: np.ndarray, ring: np.ndarray) -> bool:
    """Does `p` sit *inside* the ring of points `ring` — with them all around it?

    The test is angular rather than geometric because that is the claim being
    made: an item nobody would ask about is one with a Good in every direction
    from it. Sort the bearings to each of `ring` and ask whether any gap
    between consecutive bearings reaches half a turn; if one does, every ring
    point lies within one half-plane and `p` is outside them rather than among
    them. (For points in general position this is exactly "inside the convex
    hull", and it needs no hull.)
    """
    bearing = np.sort(np.arctan2(*(ring - p).T[::-1]))
    gaps = np.diff(np.concatenate([bearing, bearing[:1] + 2 * np.pi]))
    return bool(gaps.max() < np.pi)


def _obvious_spot(pts: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Where to draw the one item the slide calls an *obvious* match.

    The roomiest point that has a Good vote in every direction from it: sample
    the box the Goods span, keep what `_surrounded` accepts, and take whichever
    candidate is furthest from the nearest thing already drawn.

    This is a *placed* item — placed outright, where `_scene` places the item
    the app asks about by pinning it and re-fitting until it stays put. It
    exists because the honest alternative does not: the stage
    wants an unlabeled item the detector is already sure about, and picking the
    unlabeled item furthest inside the boundary — which is what this did — is
    not the same claim at all. On an elongated loop the deepest point by
    distance-to-curve sits out along the minor axis, and the one it chose sat
    **south of all five Good votes** with nothing but hollow circles beneath it:
    an item a reasonable person would want checked, captioned as one nobody
    needs to ask about (#3763). Meanwhile the middle of the votes — the one
    place on the slide where "obviously" is beyond argument — was empty, because
    the field is blue noise and the five votes are spread across it by design:
    no unlabeled item sits inside their hull at all. So the figure draws one
    there, exactly as it places the item the app asks about, and for the same
    reason: where a dot goes is the one thing this drawing is free to choose.
    """
    lo, hi = ring.min(axis=0), ring.max(axis=0)
    axes = [np.linspace(a, b, 90) for a, b in zip(lo, hi)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    inside = np.array([_surrounded(p, ring) for p in grid])
    if not inside.any():
        raise SystemExit("no point is surrounded by the Good votes — they are not a ring")
    room = np.hypot(*(pts[:, None, :] - grid[None, :, :]).transpose(2, 0, 1)).min(axis=0)
    room[~inside] = -1.0
    return grid[int(np.argmax(room))]


@functools.lru_cache(maxsize=1)
def _field() -> np.ndarray:
    """The blue-noise field, plus the obvious match the figure draws on purpose.

    Appended rather than mixed in, so every index into the field is the index
    it was before this item existed — and, more to the point, so `_seed_votes`
    (which runs on the noise alone) cannot pick it. Seeding a vote on an item
    placed at the middle of the votes would be circular, and would silently
    restart the whole story from five different items.
    """
    base = _blue_noise()
    good, _bad = _seed_votes()
    return np.vstack([base, _obvious_spot(base, base[list(good)])])


def _obvious() -> int:
    """The index of that placed item — the last one in the field."""
    return len(_blue_noise())


#: The concept the user is actually looking for, as the ground truth the field
#: was drawn against: items inside this disc are matches. Nothing in the figure
#: draws it — the whole point is that only the user knows where it is. Placed
#: right of centre so that the detector's curve, which wraps it, is nowhere
#: near the title reserve in the opposite corner.
TRUE_CENTRE = np.array([10.7, 4.25])
TRUE_RADIUS = 1.65


def _matches(pts: np.ndarray) -> np.ndarray:
    """Which of `pts` fall inside the concept."""
    return np.hypot(*(pts - TRUE_CENTRE).T) < TRUE_RADIUS


@functools.lru_cache(maxsize=1)
def _truth() -> np.ndarray:
    """Which items are matches — decided once, on the field as first drawn.

    Read off the concept disc, and then *frozen*, because an item either is or
    is not what the user is looking for and where the figure chooses to draw it
    has no say in that. Recomputing it from the settled positions instead makes
    the drawing's one arbitrary choice — where to put a dot — silently relabel
    the data it is a drawing of.
    """
    return _matches(_field())


def _spread(pts: np.ndarray, among: np.ndarray, count: int) -> tuple[int, ...]:
    """`count` of `among`, chosen to be as far apart from each other as possible.

    Farthest-point sampling, seeded on the candidate nearest the group's own
    centroid, so the answer is a pure function of the field: the opening votes
    used to be a hand-written index list, which is a hostage to every change to
    the layout — renumber the field and the deck silently starts the story from
    five different items.
    """
    # Corners are excluded: farthest-point sampling loves them, and five votes
    # sitting one in each corner of the slide reads as a diagram of a layout
    # rather than as somebody's first two minutes of clicking.
    inset = 1.25
    inner = np.array([i for i in among if (inset < pts[i]).all() and (pts[i] < np.array(CANVAS) - inset).all()])
    among = inner if len(inner) >= count else among
    picked = [int(among[np.argmin(np.hypot(*(pts[among] - pts[among].mean(axis=0)).T))])]
    while len(picked) < count:
        far = np.array([min(float(np.hypot(*(pts[i] - pts[j]))) for j in picked) for i in among])
        picked.append(int(among[int(np.argmax(far))]))
    return tuple(picked)


@functools.lru_cache(maxsize=1)
def _seed_votes() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The votes already cast when the slide opens: five Good, five Bad.

    Spread across the concept and across the rest of the field respectively, so
    that the detector they imply is a smooth loop around the concept rather
    than a shape argued from three points in one corner.
    """
    # On the noise alone, for the reason `_field` gives.
    pts = _blue_noise()
    truth = _matches(pts)
    good = _spread(pts, np.flatnonzero(truth), 5)
    # Three near-misses and two from the rest of the field. Not a cosmetic
    # split: the app asks about the items it cannot call, so most of what comes
    # back Bad early is a near-miss — and the near-misses are also what stops
    # the drawn boundary ballooning off the top of the slide, because a curve
    # is only pinned where there is something to pin it to.
    # The near-misses are taken from a *ring* rather than from everything
    # inside a radius: one Bad much closer to the concept than the other two
    # pulls a dent into the boundary that nothing else on the slide explains,
    # and a dent is the first thing anyone asks about (#3246).
    reach = np.hypot(*(pts - TRUE_CENTRE).T)
    near = np.flatnonzero(~truth & (reach > TRUE_RADIUS + 1.15) & (reach < TRUE_RADIUS + 2.1))
    far = np.flatnonzero(~truth & (reach >= TRUE_RADIUS + 3.4))
    bad = _spread(pts, near, 3) + _spread(pts, far, 2)
    return good, bad


def _fit(pts: np.ndarray, good: tuple[int, ...], bad: tuple[int, ...]) -> SVC:
    """The detector, trained on the votes cast so far.

    An RBF SVM standing in for the shipped linear one — see the module
    docstring. `gamma` is fixed rather than scaled off the data so that the
    boundary's curvature is the same object before and after the extra vote;
    with `gamma="scale"` the retrain would change the kernel as well as the
    fit, and the audience would be watching two things move at once.
    """
    idx = list(good) + list(bad)
    y = [1] * len(good) + [0] * len(bad)
    model = SVC(kernel="rbf", gamma=1.0 / (2 * KERNEL_SCALE**2), C=30.0)
    model.fit(pts[idx], y)
    return model


#: How far from the answered item the drawn retrain may differ from the
#: boundary it replaces, in figure units: the new fit within `BLEND_INNER`, the
#: old curve beyond `BLEND_OUTER`, a smooth crossfade between. Both are
#: multiples of `KERNEL_SCALE`, because that is the length scale of the thing
#: being kept — the bulge is one vote's kernel, and nothing outside a couple of
#: widths of it belongs to this page.
#:
#: Set to the tightest window that costs the drawing nothing. Measured with the
#: bulge's own peak held at 1.19 units throughout, against how sharply the drawn
#: curve turns (99.5th percentile of the angle between 0.3-unit chords — the
#: curve before the vote turns 19.2°, the raw refit 28.2°):
#:
#:     inner  outer   motion >2u   motion >3u   turn
#:      1.95   3.90        0.243        0.176   28.3
#:      1.56   2.86        0.113        0.000   28.2
#:      1.43   2.60        0.075        0.003   28.2
#:      1.30   2.34        0.036        0.007   29.4
#:
#: Tightening the window costs nothing at all until 1.30/2.34, where the curve
#: starts turning harder than the refit it is made of — the crossfade becoming
#: visible as a corner, which is the one thing this must not do. One notch
#: wider than that is the answer.
BLEND_INNER = 1.1 * KERNEL_SCALE
BLEND_OUTER = 2.0 * KERNEL_SCALE


class _Blended:
    """The retrained detector near the answered item; the old one everywhere else.

    The stage this draws has one job: show the boundary reaching out to take in
    the item the user just answered Good. An honest refit does that **and**
    something else — an RBF SVM's decision function is a sum of kernels plus a
    refitted intercept, and adding a support vector rebalances the dual, so the
    far side of the loop creeps by up to a fifth of a unit for reasons nothing
    on the slide explains. Narrowing the kernel cut that to a third (see
    `KERNEL_SCALE`) and could not remove it: the residue is not a constant
    either, so it cannot be subtracted off as one — measured on the old curve,
    the new model reads −0.09 three units from the answered item and −0.00 at
    five, which one intercept correction cannot flatten (#3763).

    So the drawn curve is spliced instead. `decision_function` crossfades the
    two real fits with a smoothstep window on distance from the answered item:
    inside `BLEND_INNER` it *is* the retrained model, beyond `BLEND_OUTER` it
    *is* the model before the vote — not approximately, identically, so the far
    side of the loop is the same ink on both pages and there is nothing there to
    watch. Everything that reads a detector goes through `decision_function`, so
    the drawn curve, the loose and tight cuts taken off it, the item the app asks
    about next, and every assertion in `_scene` all apply to this and not to a
    fit the slide never shows.

    What it costs is worth naming: this is the one boundary in the figure that
    is not a plain `sklearn` contour. It is bounded by two that are, it agrees
    with one of them exactly over most of its length, and it is still checked
    against the votes like any other — but a reader who assumed "every curve
    here is a fit" would be wrong about this one.
    """

    def __init__(self, before: SVC, after: SVC, at: np.ndarray) -> None:
        self.before, self.after, self.at = before, after, np.asarray(at, dtype=float)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        reach = np.hypot(*(X - self.at).T)
        t = np.clip((reach - BLEND_INNER) / (BLEND_OUTER - BLEND_INNER), 0.0, 1.0)
        keep_new = 1.0 - t * t * (3.0 - 2.0 * t)
        before = self.before.decision_function(X)
        return before + keep_new * (self.after.decision_function(X) - before)


def _contour(model: SVC, level: float = 0.0) -> np.ndarray:
    """A level set of the detector, as a dense set of points lying on it.

    Everything the figure needs to know about the curve is *geometric* — which
    items it passes near, which it encloses with room to spare — and none of
    that is legible from the decision function, whose units are the model's
    and not the page's. So the curve is extracted once, at the level matplotlib
    would draw, and measured against in figure units.
    """
    axis_x = np.linspace(-1.5, CANVAS[0] + 1.5, 260)
    axis_y = np.linspace(-1.5, CANVAS[1] + 1.5, 200)
    xx, yy = np.meshgrid(axis_x, axis_y)
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots()
    try:
        segments = ax.contour(xx, yy, zz, levels=[level]).allsegs[0]
    finally:
        plt.close(fig)
    # Resampled fine, because every distance in this figure is measured to
    # these points rather than to the polyline between them: at the grid's own
    # spacing the measurement is short by up to half a segment, which is enough
    # to argue about at the tolerances the settling loop works to.
    dense = []
    for seg in segments:
        step = np.hypot(*np.diff(seg, axis=0).T)
        walked = np.concatenate([[0.0], np.cumsum(step)])
        if walked[-1] <= 0:
            continue
        even = np.arange(0.0, walked[-1], CONTOUR_STEP)
        dense.append(np.column_stack([np.interp(even, walked, seg[:, axis]) for axis in (0, 1)]))
    return np.vstack(dense) if dense else np.empty((0, 2))


def _gap(curve: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each item's distance to `curve`, and the point on the curve it is near."""
    if not len(curve):
        return np.full(len(pts), np.inf), np.zeros_like(pts)
    d = np.hypot(*(curve[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    nearest = np.argmin(d, axis=0)
    return d[nearest, np.arange(len(pts))], curve[nearest]


def _next_question(
    curve: np.ndarray, pts: np.ndarray, labeled: tuple[int, ...], among: np.ndarray | None = None
) -> int:
    """The unlabeled item the app would ask about next: nearest the boundary.

    This is the `Hard` selection rule, which is the one the slide is about —
    the item whose answer the detector cannot currently guess. Measured to the
    drawn curve rather than by |decision value| so that the item the figure
    singles out is the one an audience picking by eye would also point at.

    `among` narrows the candidates, and is how the drawing picks the branch it
    is going to follow: the deck shows the user answering Good, so the item it
    hands them has to be one. That is a choice about which of two true stories
    to tell, not a fudge — the item really does end up nearest the line, because
    the field is then settled around that choice.
    """
    d, _ = _gap(curve, pts)
    d[list(labeled)] = np.inf
    if among is not None:
        keep = np.full(len(pts), np.inf)
        keep[among] = 0.0
        d = d + keep
    return int(np.argmin(d))


def _band_level(model: SVC, pts: np.ndarray, votes: tuple[int, ...]) -> float:
    """How far off zero the looser and tighter cuts sit, in decision units.

    Solved so the loosened curve sits `BAND_WIDTH` off the shipped one — the
    strip has to read as a strip — and capped so neither cut can put a check
    outside or a cross inside. That cap is the whole claim of the stage: three
    different answers, none of which the votes on screen can rule out.
    """
    ceiling = BAND_MARGIN_CAP * float(np.abs(model.decision_function(pts[list(votes)])).min())
    # Far from every support vector the decision function flattens out at the
    # model's own bias. A level past that is not a bigger loop, it is no loop:
    # the whole plane clears it and the "cut" runs off the slide.
    away = np.array([[-60.0, -60.0], [CANVAS[0] + 60, CANVAS[1] + 60]])
    ceiling = min(ceiling, BAND_MARGIN_CAP * float(np.abs(model.decision_function(away)).min()))

    zero = _contour(model)
    low, high = 0.0, ceiling
    for _ in range(24):
        level = (low + high) / 2
        loose = _contour(model, -level)
        # A level the far field also clears stops being one closed loop; treat
        # it as too wide whatever its measured distance says.
        spread = float(np.inf) if not len(loose) else float(_gap(zero, loose)[0].mean())
        low, high = (level, high) if spread < BAND_WIDTH else (low, level)
    return (low + high) / 2


def _obvious_pair(curve: np.ndarray, model: SVC, pts: np.ndarray, labeled: tuple[int, ...]) -> tuple[int, int]:
    """One unlabeled item the model is already sure is a match, and one it is sure is not.

    The foil for the slide's argument, and deliberately *one of each* rather
    than a handful of one: a top-of-the-ranking item nobody needs to ask about,
    and a bottom-of-the-ranking item nobody needs to ask about either. Together
    they say the useful question is neither the most likely nor the least, and
    a halo round three high scorers cannot say that (#3246).

    The Good is the item `_obvious_spot` placed among the Good votes; the Bad is
    chosen from the far side of the field rather than by depth alone, because
    "deepest outside" on a curve that wraps one corner of the field is just the
    opposite corner, which reads as an item picked for being far away rather
    than for being obvious.

    Both halves are checked against the drawing rather than assumed, because
    both are claims the caption makes out loud: the Good has to be well inside
    the curve *and* have a vote in every direction from it, and the Bad has to
    be well outside.
    """
    d, _ = _gap(curve, pts)
    inside = model.decision_function(pts) > 0
    good = _obvious()
    assert good not in labeled, "the placed obvious match was voted on"
    assert inside[good], "the placed obvious match sits outside the boundary"
    assert d[good] >= OBVIOUS_GAP, "the placed obvious match is not clearly inside the boundary"
    assert _surrounded(pts[good], pts[list(_seed_votes()[0])]), (
        "the placed obvious match drifted out of the ring of Good votes"
    )

    # Kept off the edges of the canvas: the deepest-outside item is usually a
    # corner, and a halo half off the slide reads as a crop rather than as a
    # mark. Away from the edge it also sits among other rejects, which is what
    # makes "the model is not asking about this one either" obvious.
    inset = 1.35
    outside = np.where(~inside, d, -1.0)
    outside[list(labeled) + [good]] = -1.0
    outside[((pts < inset) | (pts > np.array(CANVAS) - inset)).any(axis=1)] = -1.0
    bad = int(np.argmax(outside))
    assert outside[bad] >= OBVIOUS_GAP, "no unlabeled item sits clearly outside the boundary"
    return good, bad


# ──────────────────────────────────────────────────────────────────────────────
# Settling the field
# ──────────────────────────────────────────────────────────────────────────────


def _side(model: SVC, p: np.ndarray) -> int:
    """+1 if `p` is inside the model's boundary, -1 if outside."""
    return 1 if model.decision_function([p])[0] > 0 else -1


def _place(model: SVC, near: np.ndarray, p: np.ndarray, distance: float, side: int | None) -> np.ndarray:
    """The point `distance` off the curve at `near`, on the wanted side of it.

    `side=None` keeps whichever side `p` is already on, which is the rule for a
    shove: moving an item across the boundary to tidy up a drawing would change
    it from a match to a non-match, and that is a different figure.
    """
    away = p - near
    length = float(np.hypot(*away))
    # An item sitting exactly on the curve has no side to be pushed to. Give it
    # one arbitrarily; the next pass has a direction to work with.
    unit = away / length if length > 1e-9 else np.array([1.0, 0.0])
    candidate = near + unit * distance
    if side is not None and _side(model, candidate) != side:
        candidate = near - unit * distance
    return candidate


def _relax(
    pts: np.ndarray,
    curves: list[tuple[SVC, np.ndarray]],
    pins: dict[int, tuple[int, int]],
    cramped: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, int]:
    """Nudge items off the curves, and off each other, once.

    `pins` maps an item to `(which curve, which side)` and is how the two items
    the figure singles out get *placed* rather than merely cleared: they are set
    to `CURVE_CLEAR` on the named side of the named curve on every pass, so the
    item the app asks about ends up demonstrably nearer the line than anything
    else, and on the outside of it — which is what makes answering it Good move
    the boundary far enough to see.

    `cramped` names items that owe every curve only `CURVE_CLEAR` — the rule
    that actually has to hold, which is that no curve is drawn through a
    circle. An item caught in the corridor where the two boundaries run half a
    unit apart cannot clear both by the full room whatever it does, and asking
    it to stops the whole field settling over one point nobody will look at.
    See `_settle`.
    """
    moved = pts.copy()
    for k, (model, curve) in enumerate(curves):
        gaps, nearest = _gap(curve, moved)
        for i in range(len(moved)):
            pinned_curve, pinned_side = pins.get(i, (None, None))
            if pinned_curve == k:
                target = _place(model, nearest[i], moved[i], _pin_gap(i, cramped), pinned_side)
                moved[i] = moved[i] + (target - moved[i]) * PIN_DAMPING
                continue
            # Everything else is shoved toward the full room for this curve,
            # even once it counts as settled at a lesser one: a cramped item is
            # one the neighbours push straight back, so a shove that stops at
            # exactly what is required settles it a hair *under* what is
            # required on the next pass. What the item actually *owes* is
            # decided in `_violators`, which is stricter for an ordinary item
            # than for one pinned to the other curve or written off as cramped.
            strict = (CURVE_ROOM, CURVE_ROOM_AFTER)[k]
            if gaps[i] < strict - SETTLED:
                target = _place(model, nearest[i], moved[i], strict + OVERSHOOT, None)
                moved[i] = moved[i] + (target - moved[i]) * SHOVE_DAMPING

    # Then the items against each other, because a shove off a curve can shove
    # two of them together — and off the title reserve, for the same reason.
    _spread_apart(moved)
    _clear_notch(moved)
    # Counted on the positions the pass *ends* on, not the ones it started
    # from: the shoves run in sequence, so a pass that fixes every violation it
    # found can still report one, and a loop waiting for a zero never gets it.
    return moved, len(_violators(moved, curves, pins, cramped))


def _violators(
    pts: np.ndarray,
    curves: list[tuple[SVC, np.ndarray]],
    pins: dict[int, tuple[int, int]],
    cramped: frozenset[int],
) -> set[int]:
    """Which items break one of the drawing's spacing rules as the field stands."""
    broken: set[int] = set()
    for k, (model, curve) in enumerate(curves):
        gaps, _ = _gap(curve, pts)
        for i in range(len(pts)):
            pinned_curve, pinned_side = pins.get(i, (None, None))
            if pinned_curve == k:
                if gaps[i] < _pin_gap(i, cramped) - SETTLED or _side(model, pts[i]) != pinned_side:
                    broken.add(i)
                continue
            room = CURVE_TOUCH if pinned_curve is not None or i in cramped else (CURVE_ROOM, CURVE_ROOM_AFTER)[k]
            if gaps[i] < room - SETTLED:
                broken.add(i)
    apart = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(apart, np.inf)
    broken.update(int(i) for i in np.flatnonzero((apart < ITEM_APART - SETTLED).any(axis=1)))
    broken.update(i for i, p in enumerate(pts) if _in_notch(p, R + 0.10))
    return broken


def _pin_gap(item: int, cramped: frozenset[int]) -> float:
    """How far a pinned item is held off the boundary it is pinned to."""
    return CURVE_TOUCH if item in cramped else CURVE_CLEAR


def _spread_apart(moved: np.ndarray) -> None:
    """Push any two items closer than `ITEM_APART` off each other, in place.

    Swept in x order so the inner loop can stop as soon as the next item is
    further away in x alone than the rule asks for in the plane.
    """
    order = np.argsort(moved[:, 0])
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = int(order[a]), int(order[b])
            offset = moved[j] - moved[i]
            if offset[0] > ITEM_APART:
                break
            distance = float(np.hypot(*offset))
            if distance >= ITEM_APART - SETTLED:
                continue
            step = (offset / max(distance, 1e-9)) * (ITEM_APART + OVERSHOOT - distance) / 2 * SHOVE_DAMPING
            moved[i] -= step
            moved[j] += step


def _clear_notch(moved: np.ndarray) -> None:
    """Push any item that has drifted into the slide's title reserve back out.

    Out the short way — left past the reserve's left edge, or down past its
    bottom, whichever is nearer — so an item leaves the corner by the route
    that moves it least.
    """
    x0, _y0, _x1, y1 = _notch_rect()
    for i in range(len(moved)):
        if not _in_notch(moved[i], R + 0.10):
            continue
        if moved[i][1] - y1 > x0 - moved[i][0]:
            moved[i][1] = y1 + R + 0.12
        else:
            moved[i][0] = x0 - R - 0.12


#: How far the retrained boundary has to move somewhere along its length, in
#: figure units, for the audience to see that it moved at all. About 28 slide
#: pixels — a gap you can point at from the back of the room, and the whole
#: reason the stage exists. Measured as a displacement rather than as a change
#: in enclosed area: one vote bulges a short arc of a long curve, which is a
#: rounding error as an area and unmistakable as a gap.
BOUNDARY_SHIFT = 0.35


def _shift(before: np.ndarray, after: np.ndarray) -> float:
    """The furthest the retrained boundary gets from the one it replaced."""
    return float(_gap(before, after)[0].max())


#: Passes of the settling loop before an item that still cannot clear both
#: boundaries is written off as cramped and asked only to stay out of them.
SETTLE_PATIENCE = 220
#: How many items may end up cramped before the field is simply too dense to
#: draw this way and the generator should say so rather than quietly producing
#: a picture with the boundary grazing a circle.
CRAMPED_LIMIT = 6


def _settle(
    pts: np.ndarray,
    seed_good: tuple[int, ...],
    seed_bad: tuple[int, ...],
    asked: int,
    pins: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, SVC, SVC, np.ndarray, np.ndarray]:
    """Relax the field until nothing is touched, refitting as it moves.

    Two phases, because the constraints are not all equally load-bearing. Every
    item owes both boundaries `CURVE_ROOM`; an item in the corridor where the
    two run close together may not be able to pay, and one such item is enough
    to keep the whole field oscillating forever a few violations short. So
    after `SETTLE_PATIENCE` passes the still-violating items are demoted to
    owing only `CURVE_CLEAR` — no curve drawn through a circle, which is the
    rule the drawing actually turns on. The claim the extra room exists to
    protect, that the item the app asks about is the one nearest the line, is
    asserted outright by the caller rather than inferred from the spacing.
    """
    cramped: frozenset[int] = frozenset()
    for pass_number in range(3 * SETTLE_PATIENCE):
        first = _fit(pts, seed_good, seed_bad)
        # The *drawn* retrain, not the raw one — the field has to settle against
        # the curve the slide shows. See `_Blended`.
        second = _Blended(first, _fit(pts, seed_good + (asked,), seed_bad), pts[asked])
        curve, curve_after = _contour(first), _contour(second)
        pts, violations = _relax(pts, [(first, curve), (second, curve_after)], pins, cramped)
        if not violations:
            return pts, first, second, curve, curve_after
        if pass_number and pass_number % SETTLE_PATIENCE == 0:
            cramped |= frozenset(_violators(pts, [(first, curve), (second, curve_after)], pins, cramped))
            if len(cramped) > CRAMPED_LIMIT:
                raise SystemExit(f"{len(cramped)} items cannot clear both boundaries — the field is too dense")
    raise SystemExit("the field would not settle: items still touch a boundary")


@functools.lru_cache(maxsize=1)
def _scene() -> tuple[np.ndarray, SVC, SVC, np.ndarray, np.ndarray, int, int, tuple[int, ...]]:
    """The whole drawing, settled: positions, both fits, and every role.

    Settled, because the field as first drawn has the boundary running through
    half a dozen items, and an item a curve passes through reads as one the
    detector has cut in two rather than one it cannot call. Item positions are
    arbitrary — they are the one thing this figure is free to choose — so they
    are moved until nothing is touched, and the fits are redone at every step
    because moving a voted item moves the boundary that was fitted to it.

    Two items are *placed* rather than merely cleared. The item the app asks
    about is pinned just **outside** the first boundary, and the one it asks
    about next just outside the second. Outside matters: settling that item to
    wherever it happened to drift put it comfortably inside the curve, where
    the model already called it a match — so answering it Good taught the model
    nothing and the retrained boundary barely moved, which is the one thing
    this figure exists to show.

    A third is placed before the settling starts rather than during it: the
    obvious match, drawn into the middle of the Good votes by `_obvious_spot`
    (see there for why the blue noise never supplies one). It needs no pin —
    it sits a clear unit and a half from anything else and further still from
    either boundary, so no rule in `_relax` ever touches it — but `_obvious_pair`
    checks on every run that it has not drifted out of the ring it is the point
    of.

    The roles are re-derived from the settled positions in the outer pass, so
    the figure cannot end up singling out an item that was nearest the line
    before everything moved and is no longer.
    """
    pts = _field()
    seed_good, seed_bad = _seed_votes()
    labeled = seed_good + seed_bad
    matches = np.flatnonzero(_truth())
    first = _fit(pts, seed_good, seed_bad)
    asked = _next_question(_contour(first), pts, labeled, among=matches)
    second = _fit(pts, seed_good + (asked,), seed_bad)
    asked_again = _next_question(_contour(second), pts, labeled + (asked,))

    for _ in range(12):
        # Only the item the user is about to answer is pinned. The one the app
        # would ask about *next* is not: it is picked as nearest the retrained
        # boundary from the settled field, so it is nearest by construction,
        # and the question mark on it says which item it is without the drawing
        # having to win an argument about a tenth of a unit. Pinning it as well
        # over-constrains the corridor where the two boundaries run close
        # together, which is a fight no layout wins.
        pins = {asked: (0, -1)}
        pts, first, second, curve, curve_after = _settle(pts, seed_good, seed_bad, asked, pins)
        settled = (
            _next_question(curve, pts, labeled, among=matches),
            _next_question(curve_after, pts, labeled + (asked,)),
        )
        if settled == (asked, asked_again):
            break
        asked, asked_again = settled
    else:
        raise SystemExit("the item the app asks about keeps changing as the field settles")

    truth = _truth()
    assert all(truth[i] for i in seed_good), "a seeded Good sits outside the concept"
    assert not any(truth[i] for i in seed_bad), "a seeded Bad sits inside the concept"
    assert truth[asked], "the item the app asks about is not a match — the Good branch would be a lie"
    assert _side(first, pts[asked]) == -1, "the item the app asks about is already inside the boundary"
    assert _side(second, pts[asked]) == 1, "the Good vote did not bring its own item inside the new boundary"
    # The votes have to read right against the curve they trained: a check mark
    # drawn outside the blue line, or a cross drawn inside it, is a picture of
    # a detector that ignored its own training data.
    for model in (first, second):
        assert all(_side(model, pts[i]) == 1 for i in seed_good), "a Good is drawn outside the boundary"
        assert all(_side(model, pts[i]) == -1 for i in seed_bad), "a Bad is drawn inside the boundary"
    for drawn in (curve, curve_after):
        gaps, _ = _gap(drawn, pts)
        assert gaps.min() >= CURVE_CLEAR - SETTLED, "a boundary still runs through an item"
        assert not any(_in_notch(p, 0.0) for p in drawn), "a boundary runs into the title reserve"
        assert drawn[:, 0].min() > 0.3 and drawn[:, 0].max() < CANVAS[0] - 0.3, "the boundary runs off the side"
        assert drawn[:, 1].min() > 0.3 and drawn[:, 1].max() < CANVAS[1] - 0.3, "the boundary runs off the top"
    gaps, _ = _gap(curve, pts)
    gaps[list(labeled)] = np.inf
    assert int(np.argmin(gaps)) == asked, "some other item ended up nearer the line than the one it asks about"
    after, _ = _gap(curve_after, pts)
    after[list(labeled) + [asked]] = np.inf
    assert int(np.argmin(after)) == asked_again, "the next question is not the item nearest the retrained line"
    shift = _shift(curve, curve_after)
    assert shift >= BOUNDARY_SHIFT, f"the retrained boundary barely moved ({shift:.2f} units) — nothing to see"
    # And the other half of the same claim: everywhere the vote does not reach,
    # the retrained curve is not merely close to the one before it but *is* it.
    # A reveal adds ink and moves nothing else (`slides/STYLE.md`); this is that
    # rule applied to the one page of the build that redraws a curve.
    reach = np.hypot(*(curve_after - pts[asked]).T)
    away = curve_after[reach >= BLEND_OUTER]
    assert len(away), "the blend window swallowed the whole boundary — nothing is held fixed"
    assert _gap(curve, away)[0].max() <= CONTOUR_STEP, "the retrained curve leaves the old one outside the blend window"
    assert not any(_in_notch(p, R) for p in pts), "an item sits in the slide's title reserve"
    # The looser and the tighter cut have to be genuine alternatives: if either
    # of them misreads a vote, the votes *can* choose between them and the
    # slide's claim — that this is the user's call and not the data's — is
    # false. And there has to be something in the strip for the choice to be
    # about. Checked on *both* detectors: the build draws the band on the
    # retrained one and then, as a flashback, on the first.
    for model, votes in ((first, labeled), (second, labeled + (asked,))):
        level = _band_level(model, pts, votes)
        for side in (-level, level):
            drawn = _contour(model, side)
            assert len(drawn), "the loosened or tightened cut is not a curve at all"
            assert drawn[:, 0].min() > 0.2 and drawn[:, 0].max() < CANVAS[0] - 0.2, "a band cut runs off the side"
            assert drawn[:, 1].min() > 0.2 and drawn[:, 1].max() < CANVAS[1] - 0.2, "a band cut runs off the top"
            assert not any(_in_notch(p, 0.0) for p in drawn), "a band cut runs into the title reserve"
        swung = np.abs(model.decision_function(pts)) < level
        assert not swung[list(votes)].any(), "a vote falls inside the band — the two cuts are not equally defensible"
        swung[list(votes)] = False
        assert swung.sum() >= 5, f"only {swung.sum()} unlabeled items change hands between the two cuts"
    return pts, first, second, curve, curve_after, asked, asked_again, labeled


# ──────────────────────────────────────────────────────────────────────────────
# Glyphs
# ──────────────────────────────────────────────────────────────────────────────


#: Type size of the "?" an item under consideration carries.
QUERY_PT = 13


@functools.lru_cache(maxsize=1)
def _query_drop() -> float:
    """How far above its baseline to set the "?" so its ink straddles the centre."""
    box = TextPath((0, 0), "?", size=QUERY_PT, prop=FontProperties(family="DejaVu Sans", weight="bold")).get_extents()
    return -(box.y0 + box.y1) / 2 / UNIT_PT


def _circle(ax: plt.Axes, p: np.ndarray, *, asking: bool = False) -> None:
    """An item: a hollow circle, with a question mark in it while it is being asked.

    The item under consideration used to be filled solid black. Nothing else in
    the deck is solid black, so it read as a different kind of object rather
    than as the same object in a different state — and "the app is asking about
    this one" is exactly what a question mark says without a legend (#3246).
    """
    ax.add_patch(
        plt.Circle(
            tuple(p),
            R,
            facecolor="white" if asking else "none",
            edgecolor=INK,
            linewidth=2.6 if asking else 1.7,
            zorder=5 if asking else 3,
        )
    )
    if asking:
        # `va="center"` centres the font's *box*, not the glyph, and a "?" has
        # no descender to fill the bottom of that box — so it sat visibly high
        # in a circle it is supposed to be centred in (#3301). Measured off the
        # outline instead and set from its baseline.
        ax.text(
            p[0],
            p[1] + _query_drop(),
            "?",
            color=INK,
            fontsize=QUERY_PT,
            fontweight="bold",
            ha="center",
            va="baseline",
            zorder=6,
        )


def _check(ax: plt.Axes, p: np.ndarray) -> None:
    x, y = p
    ax.plot(
        [x - R, x - 0.28 * R, x + R],
        [y + 0.05 * R, y - 0.85 * R, y + R],
        color=GREEN,
        linewidth=3.4,
        solid_capstyle="round",
        solid_joinstyle="miter",
        zorder=6,
    )


def _cross(ax: plt.Axes, p: np.ndarray) -> None:
    x, y = p
    for dx in (-1, 1):
        ax.plot(
            [x - dx * R, x + dx * R],
            [y - R, y + R],
            color=RED,
            linewidth=3.2,
            solid_capstyle="round",
            zorder=6,
        )


def _grid(model: SVC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(np.linspace(-1.5, CANVAS[0] + 1.5, 420), np.linspace(-1.5, CANVAS[1] + 1.5, 300))
    return xx, yy, model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


def _boundary(ax: plt.Axes, model: SVC, *, ghost: bool = False) -> None:
    """The detector, drawn as the closed curve around what it calls a match."""
    xx, yy, zz = _grid(model)
    ax.contour(
        xx,
        yy,
        zz,
        levels=[0.0],
        colors=[GHOST if ghost else BLUE],
        linewidths=2.0 if ghost else 3.0,
        linestyles="--" if ghost else "-",
        zorder=1 if ghost else 2,
    )


def _band(ax: plt.Axes, model: SVC, level: float) -> None:
    """The same detector, cut looser and cut tighter.

    Two more level sets of the one decision function, so they are *parallel* to
    the shipped curve in the only sense that matters: they are the same
    detector read at a different threshold, not a different detector. The strip
    between them is shaded, because what the slide is asking the room to look
    at is the items inside it — every one of them is admitted by the loose cut
    and rejected by the tight one, and no vote on screen can say which is
    right.
    """
    xx, yy, zz = _grid(model)
    ax.contourf(xx, yy, zz, levels=[-level, level], colors=[BAND], zorder=0)
    ax.contour(xx, yy, zz, levels=[-level, level], colors=[BLUE], linewidths=1.6, linestyles=[(0, (5, 4))], zorder=1)


def _halo(ax: plt.Axes, p: np.ndarray) -> None:
    """A second, wider ring: "the detector is already sure about this one".

    Drawn rather than written because the figure carries no text at all beyond
    the one question mark — the presenter narrates it, which is what keeps
    every mark on the field at a size the back row can resolve.
    """
    ax.add_patch(plt.Circle(tuple(p), R + 0.20, facecolor="none", edgecolor=SOFT, linewidth=1.6, zorder=2))
    ax.add_patch(plt.Circle(tuple(p), R + 0.38, facecolor="none", edgecolor=SOFT, linewidth=1.1, zorder=2))


# ──────────────────────────────────────────────────────────────────────────────
# The figure
# ──────────────────────────────────────────────────────────────────────────────


def vote_boundary_fig() -> None:
    """One line, both of its jobs, drawn in the space the items live in.

    Nine pages. The first seven are the loop, in order: the corpus; the votes
    so far; the detector they imply; the two items nobody should be asked
    about; the one item that is worth a question, which is what the threshold
    decides for the loop; that item answered and the boundary redrawn; and the
    next question, which exists only because the boundary moved.

    The last two are the threshold's *other* job, deliberately held back to the
    end (#3254). Page 8 cuts the retrained detector looser and tighter — three
    concentric curves, all of them consistent with every vote on screen, so
    what comes back is still the user's call and not the data's. Page 9 then
    goes back and draws the same three cuts on the *first* detector, at the
    moment it was choosing what to ask. That is the point the pair exists to
    make: a threshold is not only how you cut now, it is which questions got
    you here.
    """
    for stage in range(1, VOTE_BOUNDARY_STAGES):
        save(
            _vote_boundary_stage(stage),
            OUT,
            f"vote-boundary.build{stage}.png",
            column=FULL_BLEED,
            tight=False,
            notch=VOTE_NOTCH_PX,
        )
    save(
        _vote_boundary_stage(VOTE_BOUNDARY_STAGES),
        OUT,
        "vote-boundary.png",
        column=FULL_BLEED,
        tight=False,
        notch=VOTE_NOTCH_PX,
    )


def _vote_boundary_stage(stage: int) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative).

    `FLASHBACK_STAGE` is the one page that is not cumulative: it re-draws step
    `FLASHBACK_OF` and adds the band to *that* detector, so `step` below is
    what the page shows and `stage` is only which page it is.
    """
    pts, first, second, curve, _curve_after, asked, asked_again, labeled = _scene()
    seed_good, seed_bad = _seed_votes()
    step = FLASHBACK_OF if stage == FLASHBACK_STAGE else stage

    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ── stage 1: the corpus — one hollow circle per item, none of them known ──
    voted = {
        **({i: "good" for i in seed_good} if step >= 2 else {}),
        **({i: "bad" for i in seed_bad} if step >= 2 else {}),
        **({asked: "good"} if step >= 6 else {}),
    }
    # Page g asks the next question; page h does not. Page h has moved on from
    # *what gets asked* to *what comes back*: it is the same detector cut two
    # ways, and both cuts agree with every vote on screen, so there is nothing
    # being asked on it. Page i is the one that puts the question back, because
    # its whole point is that those cuts were on offer when the question was
    # picked (#3301).
    asking = {asked} if step == 5 else ({asked_again} if step == 7 else set())

    # ── stages 8 and 9: the same detector, cut looser and cut tighter ─────────
    # Drawn under the items, and only on the two pages that are about it: page
    # 8 on the retrained detector — what comes back, now — and page 9 back on
    # the first one, where the same three cuts were already deciding what to
    # ask. Which detector the band belongs to is the whole content of the pair.
    if stage == FLASHBACK_STAGE:
        _band(ax, first, _band_level(first, pts, labeled))
    elif stage == VOTE_BOUNDARY_STAGES - 1:
        _band(ax, second, _band_level(second, pts, labeled + (asked,)))

    for i, p in enumerate(pts):
        mark = voted.get(i)
        if mark == "good":
            _check(ax, p)
        elif mark == "bad":
            _cross(ax, p)
        else:
            _circle(ax, p, asking=i in asking)

    # ── stage 3: the detector the votes imply ────────────────────────────────
    if 3 <= step <= 6:
        _boundary(ax, first, ghost=step == 6)

    # ── stage 4: the two it is already sure about — the wrong ones to ask ────
    if step == 4:
        for i in _obvious_pair(curve, first, pts, labeled):
            _halo(ax, pts[i])

    # ── stage 5: the question mark — the one item it cannot guess ────────────
    # ── stage 6: the answer, and the boundary the retrain draws instead ──────
    if step >= 6:
        _boundary(ax, second)

    # ── stage 7: which puts a different item on the new line. Repeat. ────────
    return fig


if __name__ == "__main__":
    vote_boundary_fig()
    print("wrote figures to", OUT)
