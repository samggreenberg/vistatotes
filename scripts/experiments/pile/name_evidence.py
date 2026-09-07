"""If a VG name were the only evidence, how often is the class really there?

``coco_folds.py`` answers a box question: does a VG box named *n* land on a COCO
box of class *c*? That is the right test for an **alias**, because
:func:`pilebuild.loaders.vg_scale.canonicalise` folds the box in and the band is
then read off it -- a name whose box frames something else, or frames the object
plus its cabinet, cannot serve as a positive.

It is the wrong test for the **negative pool**, and the pool is where the defect
of #3605 actually lives. An image is unusable as a negative for *c* the moment
*c* is present on it, however the box is drawn. `grandfather clock` scores 0 of
6 on box agreement -- correctly, since the box is the cabinet and COCO's is the
dial -- and COCO finds a clock on every image where it is the only evidence.

So this asks the image question instead, and asks COCO to answer it:

    over the VG-COCO overlap, take the images where VG has a box named *n*
    and **no** box named *c* -- which is exactly the situation that produces a
    false negative on the other half -- and ask COCO whether *c* is present.

That share is the name's **repair precision**. It is measured on the half with
an exhaustive reference and applied to the half without one, which is the same
trade ``anchor_to_coco`` already makes.

**Two numbers, two tables, and the split is not cosmetic:**

* ``precision`` (image level) is what :data:`pile_config.SCALE_VG_AMBIGUOUS`
  needs. Withholding an image from the pool only claims *the class may be here*.
* ``box_agree`` (box level, the same quantity ``coco_folds.py`` prints) is the
  extra thing :data:`pile_config.SCALE_VG_NAMES` needs. Folding a name in claims
  *this box is the object*, and a band is a claim about that box's size (#3616).

A name can pass the first and fail the second -- `wheel`, `clocks`,
`grandfather clock` all do -- and those are precisely the names that belong in
the ambiguous table rather than the alias one.

``base`` is the class's prevalence over the same overlap images, so a precision
can be read against the rate a name picked at random would score. A co-occurring
name inherits some of it: `wheel` is on bicycle images because bicycles have
wheels, so its precision is well above base and it is still not a bicycle.

Usage::

    python name_evidence.py --candidates cands.json --out evidence.json

``cands.json`` is ``{class: [names]}``; with no file, every class is scored
against its own head-noun family (``vg_name_families.py``).

**The verdict is derived, not drafted.** Three cuts, and each one is a number
this script measures:

* ``precision`` decides whether the name is worth acting on at all, and the
  right way to read it is as a **price**, and the units are not pool membership:
  ``1 / precision - 1`` is how many **good hard negatives are destroyed per
  contaminated negative retired**. Pool membership is not the scarce thing --
  77,119 images are eligible and only 4,200 are drawn -- but what a name
  withholds is never a random slice of them. Ranked by the class's own text
  query, `sign` takes 82% of the drawn pool's fifty hardest negatives and
  `bike` takes even more sharply, so *withholding hard negatives* is what every
  ambiguous name does and cannot discriminate between them; the ratio can.
  Measured over the 3,900-image pool (#3635): `bike` destroys 31 to retire 30,
  and `sign` destroys 435 to retire 37. The cut is on the **Wilson lower bound**, so
  a name with four supporting images does not outrank one with four hundred.
* ``box_agree`` then decides which table. Above ``--min-box`` the name's box is
  the object, so it can be folded and banded: **alias**. Below it the class is
  there but this box is not it: **ambiguous**.
* Below ``--context-box`` the box is not the object at all -- `beak`, `stop`,
  `bookshelf` -- and the name is scored **context**. Identical treatment to
  ambiguous, and reported apart because the two differ in what they cost: a
  spelling withholds the images that spell the class oddly, while a context name
  withholds a whole scene type from **every** class's pool, not just its own.
  ``--include-context`` puts them in the proposal; the default leaves them out.

**Pooled mode (``--pooled``), and why a name alone is often the wrong unit.**

The floor above is right for a name judged on its own -- 2 of 2 is not a rate --
but it left **76 of 626 candidates `unmeasured`** in #3618, carrying 312
non-COCO images that were neither acted on nor refuted. Most of those names are
not independent hypotheses. `blue umbrella`, `red umbrella`, `green umbrella`,
`orange umbrella` and `yellow umbrella` are one hypothesis five times over --
*a colour word in front of the class name does not change what the name
denotes* -- and that is testable at the sample size of the whole family.

``--pooled`` adjudicates the **group** and lets its members inherit the verdict.
The grouping is where the judgement is, so it is declared in ``pile_config`` next
to the tables it fills (:data:`pile_config.SCALE_VG_CONSTRUCTIONS` for the
productive constructions, :data:`pile_config.SCALE_VG_GROUPS` for the
hand-declared ones) and never inferred from the name at run time.

Three things stop this from being the mechanical head-noun fold #3618 refuted,
where `hot dog` (405 images, 0 of 181) rides in with `puppy`:

* **The group is counted over images, not summed over names.** An image with
  `red umbrella` and `blue umbrella` on it is one adjudicable image, not two.
* **A group must be homogeneous before it is pooled at all.** Over the members
  that clear ``--min-sole`` on their own, no member's Wilson interval may
  exclude the group's pooled rate -- at a Bonferroni-adjusted level, so a group
  with ten measured members is not condemned by multiplicity. A group that fails
  is scored ``heterogeneous`` and yields nothing: its members do not agree, so
  it was never one hypothesis. This is the gate that a group fitted to its own
  answer cannot pass, and it is why every candidate meeting a group's criterion
  is listed in the config -- winners and losers alike.
* **An individual measurement always wins.** Only a member below ``--min-sole``
  -- one with no verdict of its own -- inherits. `bike` stays ambiguous and
  `crane` stays refuted whatever their groups say. For an inherited *alias* the
  veto runs on the box axis too: a member whose own box agreement is
  significantly below its group's inherits ``ambiguous`` instead, since folding
  is the claim that costs a mis-banded positive.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import pile_config as pc

pc.setup_env()

import coco_folds as cf  # noqa: E402  (setup_env must run before vtscore resolves)

VG_ROOT = pc.DEMO_CACHE / "visual_genome"


def log(msg: str) -> None:
    print(f"[evidence] {msg}", flush=True)


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    """Lower end of the Wilson interval for *hits* of *n*.

    Used instead of the raw rate so that one cut serves names measured on five
    images and names measured on two thousand. `dove` is 5 of 5 and `bike` is
    508 of 1088; the raw rates say the first is twice the second, and the bound
    says what each one actually supports.
    """
    if n <= 0:
        return 0.0
    p = hits / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z2 / n))


def wilson_interval(hits: int, n: int, z: float) -> tuple[float, float]:
    """Both ends of the Wilson interval for *hits* of *n*, at width *z*.

    The two-sided form of :func:`wilson_lower`, used by the homogeneity gate:
    a member dissents from its group when the group's pooled rate falls outside
    this interval.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def bonferroni_z(k: int, alpha: float) -> float:
    """Two-sided *z* for *k* simultaneous intervals at family-wise level *alpha*.

    Without this a ten-member group fires the homogeneity gate 40% of the time
    on nothing at all, and the gate would read as "large groups are never one
    hypothesis" -- which is a property of the test, not of VG.
    """
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * max(k, 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="", help="JSON {class: [names]}; default = head-noun families")
    ap.add_argument("--families", default="", help="vg_name_families.py --out JSON, used when --candidates is absent")
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"))
    ap.add_argument("--iou", type=float, default=0.5, help="IoU above which two boxes are the same object")
    ap.add_argument("--min-sole", type=int, default=5, help="below this many adjudicable images a rate is not a rate")
    ap.add_argument(
        "--min-precision",
        type=float,
        default=1.0 / 3.0,
        help="Wilson lower bound a name must clear to be worth acting on. The default 1/3 is a "
        "ceiling of three images withheld from the negative pool per contaminated negative removed.",
    )
    ap.add_argument("--min-box", type=float, default=0.5, help="box agreement at or above which a name can be folded")
    ap.add_argument(
        "--min-boxes",
        type=int,
        default=20,
        help="boxes a name needs before it may be folded. Higher than --min-sole on purpose: an "
        "alias claims that EVERY box under this name is the object, and five boxes cannot carry "
        "that claim. A name below the floor falls to the ambiguous table, which is the safe side -- "
        "a wrong ambiguous costs a few pool images, a wrong alias injects a mis-banded positive.",
    )
    ap.add_argument("--context-box", type=float, default=0.1, help="below this the name is not the object at all")
    ap.add_argument("--include-context", action="store_true", help="put `context` names in the proposal too")
    ap.add_argument(
        "--pooled",
        action="store_true",
        help="adjudicate the groups declared in pile_config (SCALE_VG_CONSTRUCTIONS, SCALE_VG_GROUPS) "
        "and let a name with no verdict of its own inherit its group's (#3636)",
    )
    ap.add_argument(
        "--min-group-members",
        type=int,
        default=2,
        help="a group needs this many candidate names before it is pooled at all. Two, because a "
        "one-member group is not pooling anything -- it is the same measurement under another "
        "name, with the --min-sole floor bypassed, which is precisely what the floor is for.",
    )
    ap.add_argument(
        "--homogeneity-alpha",
        type=float,
        default=0.05,
        help="family-wise level for the gate that refuses to pool a group whose measured members "
        "disagree. Bonferroni-adjusted across those members.",
    )
    ap.add_argument("--propose-out", default="", help="write the derived tables as a name_coverage.py proposal")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    classes = list(pc.SCALE_CLASSES)
    if args.candidates:
        cands = {c: [n.strip().lower() for n in ns] for c, ns in json.loads(Path(args.candidates).read_text()).items()}
    elif args.families:
        fam = json.loads(Path(args.families).read_text())
        cands = {c: [r["name"] for r in rows] for c, rows in fam["families"].items()}
    else:
        raise SystemExit("pass --candidates or --families")
    unknown = set(cands) - set(classes)
    if unknown:
        raise SystemExit(f"candidates name classes that are not in C: {sorted(unknown)}")

    wanted = set(classes) | {n for ns in cands.values() for n in ns}
    log(f"{len(classes)} classes; {len(wanted)} VG names")

    # Which pooled groups each class's candidates fall into. Resolved once,
    # from the declarations in pile_config, so the objects.json pass can count
    # a group at the image level rather than summing its members afterwards.
    groups: dict[str, dict[str, list[str]]] = {}
    if args.pooled:
        groups = {c: pc.scale_vg_groups_for(c, list(names)) for c, names in cands.items()}
        groups = {c: {g: m for g, m in gs.items() if len(m) >= args.min_group_members} for c, gs in groups.items()}
        groups = {c: g for c, g in groups.items() if g}
        n_groups = sum(len(g) for g in groups.values())
        log(f"pooled: {n_groups} groups over {len(groups)} classes")

    cboxes, cdims, cpresent = cf.coco_boxes(Path(args.anchor_dir))

    log("loading VG image_data.json")
    with (Path(args.anchor_dir) / "image_data.json").open() as fh:
        meta = json.load(fh)
    coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}
    vdims = {int(m["image_id"]): (int(m["width"]), int(m["height"])) for m in meta}

    log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)
    log(f"  {len(records)} VG records")

    # per (class, name): the image question, adjudicated by COCO
    sole = defaultdict(int)  # overlap images with a box named n and none named c
    sole_hit = defaultdict(int)  # ... on which COCO says c is present anyway
    # per (class, name): the box question -- n's boxes that land on a COCO c box
    boxes = defaultdict(int)
    boxes_hit = defaultdict(int)
    # supply, and the images the fold would actually act on
    vg_images = defaultdict(int)
    off_sole = defaultdict(int)
    # the same four questions asked of a GROUP, counted over IMAGES: an image
    # carrying `red umbrella` and `blue umbrella` is one adjudicable image, and
    # summing the members would count it twice and halve the interval.
    gsole = defaultdict(int)
    gsole_hit = defaultdict(int)
    gboxes = defaultdict(int)
    gboxes_hit = defaultdict(int)
    goff_sole = defaultdict(int)
    # base rate: overlap images where COCO annotates c at all
    overlap_images = 0
    base_hit = dict.fromkeys(classes, 0)

    #: The COCO classes whose annotation counts as *c*, resolved once.
    #:
    #: This is the difference between asking about the class and asking about
    #: the COCO class of the same name, and for a class this project defines as
    #: a UNION (:data:`pile_config.SCALE_CLASS_MERGES`) those are not the same
    #: question. `cup` is `cup` U `wine glass`, and scoring a stemware spelling
    #: against COCO `cup` alone reads its boxes as landing on nothing: `wine
    #: glass` measured 38% precision and **2%** box agreement that way, a
    #: `neither` verdict produced entirely by the scorer looking at the wrong
    #: half. `mug`, whose object COCO really does call a cup, scored normally --
    #: which is what made the defect specific to the merged class and silent
    #: everywhere else (#3700).
    #:
    #: :func:`pile_config.coco_classes_for` returns ``{c}`` for an unmerged
    #: class, so this changes nothing for the other twenty-four.
    coco_for = {c: pc.coco_classes_for(c) for c in classes}

    skipped_aspect = 0
    for rec in records:
        iid = int(rec["image_id"])
        vd = vdims.get(iid)
        if vd is None:
            continue

        by_name: dict[str, list[list[float]]] = {}
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if not names:
                continue
            name = str(names[0]).strip().lower()
            if name not in wanted:
                continue
            x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
            w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
            if w <= 0 or h <= 0:
                continue
            by_name.setdefault(name, []).append([x / vd[0], y / vd[1], (x + w) / vd[0], (y + h) / vd[1]])
        for name in by_name:
            vg_images[name] += 1

        cid = coco_of.get(iid)
        if cid is None or cid not in cdims:
            for c, names in cands.items():
                if c not in by_name:
                    for n in names:
                        if n in by_name:
                            off_sole[c, n] += 1
                    for g, members in groups.get(c, {}).items():
                        if any(m in by_name for m in members):
                            goff_sole[c, g] += 1
            continue

        cd = cdims[cid]
        if abs((vd[0] / vd[1]) - (cd[0] / cd[1])) > pc.MAX_ASPECT_DRIFT:
            skipped_aspect += 1
            continue
        overlap_images += 1
        here = cpresent.get(cid, set())
        for c in classes:
            base_hit[c] += bool(here & coco_for[c])
        for c, names in cands.items():
            # Both the presence question and the box question are asked of the
            # class's WHOLE COCO footprint, not of its namesake alone (#3700).
            truth = [b for k in coco_for[c] for b in cboxes.get(cid, {}).get(k, [])]
            for n in names:
                vb = by_name.get(n)
                if not vb:
                    continue
                if c not in by_name:
                    # exactly the state that becomes a false negative off-COCO
                    sole[c, n] += 1
                    sole_hit[c, n] += bool(here & coco_for[c])
                boxes[c, n] += len(vb)
                boxes_hit[c, n] += sum(1 for b in vb if any(cf.iou(t, b) >= args.iou for t in truth))
            for g, members in groups.get(c, {}).items():
                present = [m for m in members if m in by_name]
                if not present:
                    continue
                if c not in by_name:
                    gsole[c, g] += 1
                    gsole_hit[c, g] += bool(here & coco_for[c])
                for m in present:
                    gboxes[c, g] += len(by_name[m])
                    gboxes_hit[c, g] += sum(1 for b in by_name[m] if any(cf.iou(t, b) >= args.iou for t in truth))

    log(f"{overlap_images} adjudicable overlap images; skipped {skipped_aspect} on aspect drift")

    def pct(a: int, b: int) -> str:
        return f"{100.0 * a / b:.0f}%" if b else "--"

    def verdict(c: str, n: str) -> str:
        """Which table *n* belongs in for *c*, from the three cuts above."""
        if sole[c, n] < args.min_sole:
            return "unmeasured"
        if wilson_lower(sole_hit[c, n], sole[c, n]) < args.min_precision:
            return "neither"
        if boxes[c, n] < args.min_boxes:
            # Precision cleared, so the class is there; there are just not enough
            # boxes to claim any of them IS it. Fall to the safe table.
            return "ambiguous"
        agree = boxes_hit[c, n] / boxes[c, n]
        if agree >= args.min_box:
            return "alias"
        return "ambiguous" if agree >= args.context_box else "context"

    verdicts = {(c, n): verdict(c, n) for c, names in cands.items() for n in names if vg_images[n]}

    # ---- pooled adjudication (#3636) -------------------------------------
    # A group is scored with the SAME three cuts as a name, over its own
    # image-level counts, and only then is anything inherited.
    RANK = {"alias": 2, "ambiguous": 1, "context": 1}

    def group_row(c: str, g: str, members: list[str]) -> dict:
        live = [n for n in members if vg_images[n]]
        n_sole, n_hit = gsole[c, g], gsole_hit[c, g]
        nb, nbh = gboxes[c, g], gboxes_hit[c, g]
        # The homogeneity gate, over the members that have a rate of their own.
        #
        # Its reference is the MEMBER-weighted rate, not the image-level one
        # above: the two differ whenever members share an image (counted once in
        # the union and once per member), and comparing a member against a
        # denominator it is not part of is how `paper` and `papers` both came
        # out dissenting from a pool that lies between them. The union rate stays
        # the group's verdict statistic, because the price is per image.
        m_sole = sum(sole[c, n] for n in live)
        m_hit = sum(sole_hit[c, n] for n in live)
        member_rate = m_hit / m_sole if m_sole else 0.0
        measured = [n for n in live if sole[c, n] >= args.min_sole]
        z = bonferroni_z(len(measured), args.homogeneity_alpha)
        dissent = []
        for n in measured:
            lo, hi = wilson_interval(sole_hit[c, n], sole[c, n], z)
            # The tolerance is not cosmetic: at p = 1 the Wilson upper end is
            # analytically 1 and evaluates to 0.9999999999999999, so a group
            # whose every member is 23 of 23 and 6 of 6 was declared
            # heterogeneous against a pooled rate of exactly 1.0.
            if not (lo - 1e-9 <= member_rate <= hi + 1e-9):
                dissent.append(n)
        foldable = pc.scale_vg_group_foldable(c, g)
        if n_sole < args.min_sole:
            v = "thin"
        elif dissent:
            v = "heterogeneous"
        elif wilson_lower(n_hit, n_sole) < args.min_precision:
            v = "neither"
        elif nb < args.min_boxes:
            v = "ambiguous"
        else:
            agree = nbh / nb
            v = "alias" if agree >= args.min_box else ("ambiguous" if agree >= args.context_box else "context")
        if v == "alias" and not foldable:
            v = "ambiguous"  # a collective can be evidence, never a band
        return {
            "members": live,
            "measured": measured,
            "dissent": dissent,
            "sole": n_sole,
            "sole_present": n_hit,
            "member_sole": m_sole,
            "member_rate": member_rate,
            "precision_lower": wilson_lower(n_hit, n_sole),
            "boxes": nb,
            "boxes_on_class": nbh,
            "off_coco_sole": goff_sole[c, g],
            "foldable": foldable,
            "why": pc.scale_vg_group_why(c, g),
            "verdict": v,
        }

    group_rows: dict[str, dict[str, dict]] = {}
    inherited: dict[tuple[str, str], str] = {}
    inherit_from: dict[tuple[str, str], list[str]] = defaultdict(list)
    conflicts: list[tuple[str, str, list[str]]] = []
    for c, gs in groups.items():
        group_rows[c] = {g: group_row(c, g, m) for g, m in gs.items()}
        # Only a name with no verdict of its own inherits. `bike` stays
        # ambiguous and `crane` stays refuted whatever their groups say.
        claims: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for g, row in group_rows[c].items():
            if row["verdict"] not in RANK:
                continue
            zb = bonferroni_z(len(row["members"]), args.homogeneity_alpha)
            pooled_box = row["boxes_on_class"] / row["boxes"] if row["boxes"] else 0.0
            for n in row["members"]:
                if sole[c, n] >= args.min_sole:
                    continue
                v = row["verdict"]
                if v == "alias" and boxes[c, n]:
                    # The veto on the axis folding actually risks: a member
                    # whose own boxes land on the class significantly less
                    # often than its group's is evidence, not a band.
                    _, hi = wilson_interval(boxes_hit[c, n], boxes[c, n], zb)
                    if hi < pooled_box:
                        v = "ambiguous"
                claims[n].append((g, v))
        for n, got in claims.items():
            best = min(got, key=lambda t: RANK[t[1]])
            if len({v for _, v in got}) > 1:
                conflicts.append((c, n, [f"{g}={v}" for g, v in got]))
            inherited[c, n] = best[1]
            inherit_from[c, n] = [g for g, _ in got]

    print("\n" + "=" * 104)
    print("NAME EVIDENCE -- COCO adjudicating the images where a name is the class's ONLY evidence")
    print("`sole` = overlap images with the name and not the class name. `precision` = COCO says the")
    print("class is present anyway. `box` = the name's boxes landing on a COCO box of the class.")
    print(f"A row with fewer than {args.min_sole} sole images is marked `thin`: it is a count, not a rate.")
    print("=" * 104)
    for c in classes:
        if c not in cands:
            continue
        base = 100.0 * base_hit[c] / overlap_images if overlap_images else 0.0
        print(f"\n{c}   (base rate {base:.1f}% of overlap images; {vg_images[c]} VG images under the class name)")
        print(
            f"    {'name':<26}{'VG imgs':>8}{'sole':>7}{'prec':>6}{'lower':>7}"
            f"{'boxes':>7}{'box':>6}{'off-COCO':>10}  verdict"
        )
        rows = sorted(cands[c], key=lambda n: -off_sole[c, n])
        residual = [0, 0, 0]
        for n in rows:
            if not vg_images[n]:
                continue
            v = verdicts[c, n]
            got = inherited.get((c, n))
            if v == "unmeasured" and not got:
                residual[0] += sole[c, n]
                residual[1] += sole_hit[c, n]
                residual[2] += off_sole[c, n]
            if v == "neither" and off_sole[c, n] < 20:
                continue  # a refuted name with no supply is noise in a long table
            lower = wilson_lower(sole_hit[c, n], sole[c, n])
            shown = f"{got} <- {'+'.join(inherit_from[c, n])}" if got else v
            print(
                f"    {n:<26}{vg_images[n]:>8}{sole[c, n]:>7}{pct(sole_hit[c, n], sole[c, n]):>6}"
                f"{lower:>7.2f}{boxes[c, n]:>7}{pct(boxes_hit[c, n], boxes[c, n]):>6}"
                f"{off_sole[c, n]:>10}  {shown}"
            )
        # What the sole-image floor leaves behind, pooled: the same question
        # asked of every name too thin to answer it alone.
        if residual[0]:
            print(
                f"    {'(still unmeasured)':<26}{'':>8}{residual[0]:>7}{pct(residual[1], residual[0]):>6}"
                f"{wilson_lower(residual[1], residual[0]):>7.2f}{'':>7}{'':>6}{residual[2]:>10}  residual"
            )
        for g, row in sorted(group_rows.get(c, {}).items()):
            got = [n for n in row["members"] if g in inherit_from.get((c, n), [])]
            print(
                f"    [{g}]{'':<21}{'':>8}{row['sole']:>7}{pct(row['sole_present'], row['sole']):>6}"
                f"{row['precision_lower']:>7.2f}{row['boxes']:>7}{pct(row['boxes_on_class'], row['boxes']):>6}"
                f"{row['off_coco_sole']:>10}  {row['verdict']}"
                + (f" -> {len(got)} name{'s' if len(got) != 1 else ''}" if got else "")
            )

    if args.pooled:
        print("\n" + "=" * 104)
        print("POOLED GROUPS -- a construction or a declared kind, adjudicated as ONE hypothesis (#3636)")
        print("`sole` counts IMAGES, so an image carrying two members of a group is one adjudicable image.")
        print("`dissent` is a measured member whose own rate excludes the pooled one: the group is then not")
        print("one hypothesis and yields nothing. Only a member below the sole floor ever inherits.")
        print("=" * 104)
        for c in classes:
            if c not in group_rows:
                continue
            print(f"\n{c}")
            for g, row in sorted(group_rows[c].items()):
                # Only the names this group actually granted a verdict to. Keyed
                # on `inherited` alone, a heterogeneous group listed names that
                # a sibling group had settled -- which reads as the gate failing.
                got = [n for n in row["members"] if g in inherit_from.get((c, n), [])]
                print(
                    f"    [{g}]  {row['why']}\n"
                    f"        {len(row['members'])} names"
                    f"{'' if row['foldable'] else ' (never foldable)'}"
                    f"; sole {row['sole']} at {pct(row['sole_present'], row['sole'])}"
                    f" (lower {row['precision_lower']:.2f});"
                    f" boxes {row['boxes']} at {pct(row['boxes_on_class'], row['boxes'])}"
                    f"; off-COCO {row['off_coco_sole']}  ->  {row['verdict'].upper()}"
                )
                print(f"        members: {' '.join(row['members'])}")
                if row["dissent"]:
                    print(f"        DISSENT: {' '.join(row['dissent'])}  (measured, and disagree with the pool)")
                if got:
                    print(f"        inherited by: {' '.join(f'{n}={inherited[c, n]}' for n in sorted(got))}")
        n_inh = len(inherited)
        n_off = sum(off_sole[c, n] for c, n in inherited)
        print(f"\n{n_inh} names inherited a verdict, carrying {n_off} non-COCO images.")
        if conflicts:
            print("Groups disagreed on these names; the most conservative verdict was taken:")
            for c, n, got in conflicts:
                print(f"    {c} / {n}: {', '.join(got)} -> {inherited[c, n]}")

    def final(c: str, n: str) -> str:
        """The verdict that reaches the tables: a name's own, else its group's."""
        return inherited.get((c, n), verdicts.get((c, n), "unmeasured"))

    if args.propose_out:
        keep = {"ambiguous", "context"} if args.include_context else {"ambiguous"}
        proposal = {
            "alias": {
                c: sorted(n for n in names if final(c, n) == "alias")
                for c, names in cands.items()
                if any(final(c, n) == "alias" for n in names)
            },
            "ambiguous": {
                c: sorted(n for n in names if final(c, n) in keep)
                for c, names in cands.items()
                if any(final(c, n) in keep for n in names)
            },
        }
        Path(args.propose_out).write_text(json.dumps(proposal, indent=1) + "\n")
        n_alias = sum(len(v) for v in proposal["alias"].values())
        n_ambig = sum(len(v) for v in proposal["ambiguous"].values())
        print(f"\nwrote {args.propose_out}: {n_alias} alias names, {n_ambig} ambiguous names")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "meta": {
                        "iou": args.iou,
                        "overlap_images": overlap_images,
                        "skipped_aspect_drift": skipped_aspect,
                        "min_sole": args.min_sole,
                        "min_precision": args.min_precision,
                        "min_box": args.min_box,
                        "min_boxes": args.min_boxes,
                        "context_box": args.context_box,
                        "pooled": args.pooled,
                        "min_group_members": args.min_group_members,
                        "homogeneity_alpha": args.homogeneity_alpha,
                    },
                    "base_rate": {c: base_hit[c] / overlap_images if overlap_images else 0.0 for c in classes},
                    "class_images": {c: vg_images[c] for c in classes},
                    "names": {
                        c: {
                            n: {
                                "vg_images": vg_images[n],
                                "sole": sole[c, n],
                                "sole_present": sole_hit[c, n],
                                "boxes": boxes[c, n],
                                "boxes_on_class": boxes_hit[c, n],
                                "off_coco_sole": off_sole[c, n],
                                "precision_lower": wilson_lower(sole_hit[c, n], sole[c, n]),
                                "verdict": verdicts[c, n],
                                # What actually reaches the tables, and from where.
                                "final": final(c, n),
                                "inherited_from": inherit_from.get((c, n), []),
                            }
                            for n in names
                            if vg_images[n]
                        }
                        for c, names in cands.items()
                    },
                    "groups": group_rows,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
