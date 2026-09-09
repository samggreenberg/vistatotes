# 2026-09-06 — a backup glob that did not match the sibling (#3667)

**Cost:** ~0h of compute, one measurement downgraded from measured to reconstructed.

**What broke:** before rebuilding the `vg_scale` pile I copied the cells aside
with `cp $D/vg_scale__*.pkl $D/vg_scale_any__*.pkl $B/`. That is 4.7 GB and
looks exhaustive. It does not match `vg_scale_deep__siglip.pkl`, because
`vg_scale__*` requires the character after `vg_scale` to be `_` twice and the
deep cell spells it `vg_scale_d`. The deep cell was then overwritten by the
rebuild with no copy of it anywhere.

Recoverable here only by luck of construction: the pre-#3667 rule
(`evaluable = categories or (all cells if in the shared pool else [])`) inverts
using nothing the rebuilt pickle lacks, so the label counts could be
reconstructed exactly. Its membership and its vectors could not, and the
before/after comparison for that dataset is a reconstruction rather than a
measurement — labelled as such wherever it appears in the report.

**The general shape:** a glob written from the name of the thing you are
thinking about will silently miss a sibling whose name *extends* it. `vg_scale`,
`vg_scale_any` and `vg_scale_deep` are three datasets and one prefix.

**Prevented?** *Advice only.* Enumerate what you are about to overwrite and copy
*that* — `build_pile.py --list` prints every cell of every dataset — rather than
writing a pattern and trusting it. Cheap check before a destructive rebuild:
count the files the glob matched against the cells the build is about to write.
