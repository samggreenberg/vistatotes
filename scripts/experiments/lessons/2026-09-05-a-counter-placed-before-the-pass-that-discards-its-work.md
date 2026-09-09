# 2026-09-05 — a counter placed before the pass that discards its work (#3637)

**Study:** #3637, is a scattered fold the right outcome? **Cost:** none to this
study, which is the only reason it is worth writing down: the number was wrong
by a factor of two for a release, in a log line whose entire job is to be looked
at, and nothing anywhere could have said so.

`vg_scale`'s build is eight named passes. `canonicalise` folds an alias
spelling's boxes onto the class name and logs how many it folded —
`merged VG spellings: bird+607, boat+602, …` — and that line exists precisely
because *a merge that folds nothing has either been mis-spelled or is not
needed, and both are worth seeing*. It ran **first**. The pass after it,
`anchor_to_coco`, does this:

```python
labels[iid] = {name: bs for name, bs in ref.items() if name in wanted}
```

— a wholesale replacement of VG's labels with COCO's, on the 48% of VG that COCO
annotates. **Every box folded onto an anchored image is discarded one line
later.** Measured: the log reported **5,142** boxes folded where the build keeps
**2,559**. Half of every fold the line has ever reported was thrown away before
anything read it.

Nothing was *broken*. The dataset was correct, the ratio was stable, the tests
passed, and the number looked entirely plausible — a wrong number that moves the
right way is the hardest kind to see. It surfaced only because #3637 needed the
pass to report a *second* quantity (how many images the fold un-bands) and that
one cannot be computed before the anchor at all: it needs `box_dims`, which is
what `anchor_to_coco` returns.

**The generalisable rule: count where the work lands, not where it is
attempted.** A pass that reports what it did must sit downstream of everything
that can undo it. Two questions settle it in review, and neither needs a
measurement:

1. Does any later pass *overwrite* rather than *edit* what this one produced? A
   whole-dict assignment (`labels[iid] = …`) is the tell; an `.update()` or a
   `.setdefault().extend()` is not.
2. Would the counter still be right if that later pass ran on every item?

Here the fix was free, because the fold and the anchor commute on every image
either one changes: on an anchored image the fold's effect is discarded, and on
an unanchored one the anchor does nothing. So `canonicalise` moved to *after*
`anchor_to_coco` in both `vg_scale` and `vg_scale_deep`, the counter became
exact, and the new `contested` counter became computable at all.

**Still only advice — but the claim itself is now checked.** "The reorder is a
no-op" is exactly the kind of assertion that is comfortable to write and
expensive to be wrong about, so `band_fold.py`'s supply phase carries the old
pass order as a fourth arm and asserts that it designates identical image ids in
all 36 cells. That is a study-local control, not a gate: nothing stops the next
pass reorder from shipping unverified. If a second instance of this shows up, the
check to add is a build-time one — assert that no pass after a reporting pass
whole-assigns into `labels`.

Full study: [`docs/experiments/2026-09-05-band-fold-3637/`](../../../docs/experiments/2026-09-05-band-fold-3637/REPORT.md).
