# DocMarks on the GRID

The full-scale run: every source, all three tiers, embedding cells cached in the
shared pile. Read [`README.md`](README.md) first for what the corpus *is*; this
is only how to build it at size on the cluster.

A tier-`s` SPODS-only build fits comfortably on a laptop or a cloud container
(SPODS is 2.94 GB down, ~8 GB unpacked). What needs the GRID is tiers `m` and
`l`: 50k–200k UCSF pages to fetch and render, and `sift_vlad` local features
over all of them.

## Sizing

Measured or estimated per page: a rendered UCSF page at 150 dpi is ~250 KB PNG;
`sift_vlad` adds 128–256 KB of local features per image in its cell; SigLIP adds
3 KB.

| tier | pages | images on disk | `sift_vlad` cell | `siglip` cell |
|---|---:|---:|---:|---:|
| `s` | 5k | ~1.5 GB | ~1 GB | ~20 MB |
| `m` | 50k | ~13 GB | ~10 GB | ~160 MB |
| `l` | 200k | ~50 GB | ~40 GB | ~640 MB |

Plus raw sources: SPODS 2.94 GB archive → ~8 GB unpacked; the UCSF PDF cache is
roughly the same size again as the rendered pages.

**Put `VTS_DOCMARKS_RAW` and `VTS_DOCMARKS_OUT` on `/expscratch`, never on the
50 GB mount.** `GRID-PLAYBOOK.md` describes that mount as chronically full, and
tier `l` alone would fill it twice over. `df` the experiment path itself, not a
parent — the free space differs.

```bash
export VTS_DOCMARKS_RAW=/expscratch/$USER/docmarks/raw
export VTS_DOCMARKS_OUT=/expscratch/$USER/docmarks/corpus
source scripts/experiments/pile/pile_env.sh    # VTS_PILE, models, HF_HOME
```

## Preflight

```bash
python build_corpus.py --probe
```

Every source fails differently and the probe is the cheapest place to find out
which: a decommissioned hostname (SPODS's own page still advertises the dead
`ernet.in` host), a missing Kaggle token (StaVer, Tobacco800), an absent RAR
extractor, or a UCSF endpoint that is down. Seconds now, a queue slot later.

**The probe downloads nothing**, so it is safe to run repeatedly on a login
node: a `HEAD` for SPODS, a file listing for the Kaggle mirrors, a result count
for UCSF. It also sweeps away any `_probe_*` directories left under
`$VTS_DOCMARKS_RAW` by the version that *did* fetch (~2 GB of duplicated
StaVer/Tobacco800 bytes), reporting what it reclaimed.

**RAR extractor.** SPODS is RAR4. The builder tries `bsdtar`, `7z`, `unar`,
`unrar` in that order. **On this cluster none of the four is present** — not on
the login nodes and not on a compute node, so the hope that `bsdtar` would be
there did not survive contact. `libarchive.so.13` is installed but the `bsdtar`
binary that fronts it is not, so the probe reports `rar extractor: NONE FOUND`
and SPODS is unreachable, Kaggle token or no Kaggle token.

Fixed once, for the user, by dropping a static 7-Zip in `~/.local/bin`:

```bash
cd /expscratch/$USER && curl -sLO https://www.7-zip.org/a/7z2501-linux-x64.tar.xz
tar xf 7z2501-linux-x64.tar.xz 7zz && install -m 755 7zz ~/.local/bin/7zz
ln -sf ~/.local/bin/7zz ~/.local/bin/7z          # the name the builder looks for
```

It is a single static binary with no dependencies, and `7zz i` lists the `Rar3`
codec that RAR4 needs. `launch_docmarks.sh` prepends `~/.local/bin` to `PATH`
because a **login shell has that directory and an sbatch job does not** — the
probe passing interactively is not evidence the job will find the extractor.

**Kaggle token.** `~/.kaggle/kaggle.json`, `~/.kaggle/access_token`, or
`KAGGLE_USERNAME` + `KAGGLE_KEY` in the job environment. Needed only for StaVer
and Tobacco800; a SPODS-only roster does not touch Kaggle.

`access_token` is the file "Create New Token" writes today, and the one
`kagglesdk` reads; the probe accepts it as of #3343. Also note the `kaggle` CLI
itself is not installed on this cluster — `uv tool install kaggle` puts it in
`~/.local/bin` without touching the shared venv, which is what the launcher's
`PATH` already picks up.

Then the standard gate, which checks the things a script cannot:

```bash
bash scripts/experiments/preflight.sh --exp "$VTS_DOCMARKS_OUT" \
  --job-name docmarks-build --mem 16G --conc 8
```

## Stage 1 — sources and clustering (CPU, one job)

Fetching is network-bound and single-threaded per source; rendering is
CPU-bound. This is one long job rather than an array, because the sources are
sequential and the clustering needs every mark in memory at once.

```bash
python build_corpus.py \
  --sources spods,staver,tobacco800,ucsf \
  --ucsf-distractors 200000 \
  --ucsf-letterhead-per-author 2000
```

Ask for **64 GB and a wall clock in hours**, not minutes. 16 GB was the
inherited figure; a 1,541-page smoke already peaked at 6.5 GB, and clustering
holds every mark at once while the letterhead candidates go from 160 to 16,000.

**Do not parallelise the pull across many jobs.** That is the rule, and it is
about jobs: a single job fetches 3-wide behind `_Throttle`, which surrenders a
worker and doubles its delay on any 429/503/509, so pushback converges on serial
instead of hammering through. `VTS_DOCMARKS_FETCH_WORKERS=1` restores the
strictly-serial pull.

What that 3 rests on, so the next person can re-derive it rather than trust it:
~120,000 requests at ~3/s drew **zero** rate-limit responses, and the 4,003
failures in the first full build were **403 Access Denied** — PDFs indexed but
not public, permanent and unrelated to pacing. That says we were far below
UCSF's limit, *not* where the limit is; the throttle exists because those are
different claims. See
[`lessons/2026-09-01-a-caution-in-a-runbook-was-read-as-a-measured-limit.md`](../lessons/2026-09-01-a-caution-in-a-runbook-was-read-as-a-measured-limit.md).

The builder skips a dead id rather than aborting and reports the count at the
end. **Classify the skips before accepting them**: 403s are permanently
unavailable documents and are normal; a run of 429s means back off; and tens of
thousands of anything means the endpoint, not the data. The pull also prints its
own throughput and CPU share when it finishes — a pull that never backs off and
sits at 37% CPU is under-driven, which is invisible in an ETA.

**Resume is free** — measured, at ~89 pages/s against ~8.6 cold. Downloads are
atomic (temp + rename), the Solr cursor order is stable, and a page whose PNGs
are all on disk is **not re-rendered**: its dimensions are read back off the
images. That last part was false until #3343 (the skip guarded the *save* while
the render above it ran unconditionally, so a resumed job re-rendered everything
and threw it away — ~10 h at 200k pages). It is what makes a multi-day pull
correctable mid-flight rather than something you can only endure, so it is
pinned by `TestResumeSkipsRendering`.

Before resuming, delete zero-byte outputs — they count as "done" and resume
cannot see that they are empty — and any `.part` files, which resume via a Range
header that is only safe if the server honours it.

## Stage 2 — roster (interactive, off-cluster)

```bash
python shortlist.py --corpus $VTS_DOCMARKS_OUT --write-roster --name docmarks-v1
```

Copy `shortlist.png` and `roster.json` somewhere you can look at them, pick your
two dozen, edit the file, then rebuild in roster mode:

```bash
python build_corpus.py --sources spods,staver,tobacco800,ucsf --roster $VTS_DOCMARKS_OUT/roster.json
```

Rebuilding is cheap — the sources are cached, so only clustering and manifest
writing re-run.

## Stage 3 — the human passes (one bundle, then interactive)

Render both sheet sets on the cluster and take the result away as one archive:

```bash
bash launch_docmarks.sh slate           # ~minutes, CPU, no refetch
# when it finishes, the log names the bundle:
scp $GRID:$VTS_DOCMARKS_OUT/audit/docmarks-audit-<date>.tar.gz .
```

**Then the second opinion, on a GPU**, which re-renders both similarity passes
against a semantic embedder (see README's *The descriptor the audit asks with is
not the one it was built with*):

```bash
bash launch_docmarks.sh siglip          # ~minutes, GPU, embeds ~1.3k crops once
scp $GRID:$VTS_DOCMARKS_OUT/audit/docmarks-audit-siglip-<date>.tar.gz .
```

It re-numbers the slate, so a `merges.txt` written against the `phash` sheets no
longer refers to the same classes — the job keeps the old answer beside the new
one as `merges.phash.txt` rather than overwriting it. Work the `phash` slate
only if no GPU is available; the appendix it produces spends most of its 120
pairs on classes that are not confusable (#3600).

Rendering happens on the GRID because that is where the pages are: the sheets
crop from full-resolution scans that live under `$VTS_DOCMARKS_OUT`, and pulling
200k pages to a laptop to make four contact sheets is the wrong direction. The
bundle is small — a few dozen PNGs — so the *reviewing* happens wherever you
like.

**Work `audit/merge` first, then `audit/membership`.** They are one sitting, and
they answer different questions: the merge slate fixes the *partition* (which
clustering proposals are one mark) and membership fixes the *instances* (whether
each crop really is that mark). Doing them a week apart means re-deriving the
same context twice.

- `merge/slate_*.png` — every class as a numbered 3-up strip, similarity-ordered.
- `merge/pairs_*.png` — the closest class pairs, explicitly side by side.
- `merge/merges.txt` — the answer: one line per group of same-mark indices, plus
  `REVIEWED-ALL` once you have worked every sheet. See README's **The slate**.
- `membership/*.png` — every instance of every class, numbered.
- `membership/verdicts.jsonl` — `ok`, or the indices that are not this mark.

Then fold both back in, on the cluster, against the corpus they came from:

```bash
python audit_to_corrections.py --task merge      --corpus $VTS_DOCMARKS_OUT   # dry run
python audit_to_corrections.py --task merge      --corpus $VTS_DOCMARKS_OUT --apply
python audit_to_corrections.py --task membership --corpus $VTS_DOCMARKS_OUT --apply
```

Both default to a dry run; read it before passing `--apply`. Merge first: a
membership rejection is scoped to one class, and which class that is can change
under a merge.

Do this **before** stage 4. A membership rejection changes which pages are
positives, and the embedding cells carry the labels; embedding first means
rebuilding every cell afterwards.

`--task confusable` is the same adjudication one pair per sheet. It remains the
right form for a roster small enough to work the full matrix; at v2's 60 classes
that is 1,770 sheets, which is why the slate exists.

## Stage 4 — embedding cells (GPU)

```bash
python embed_corpus.py --tier s --embedders sift_vlad,siglip
python embed_corpus.py --tier m --embedders sift_vlad,siglip
python embed_corpus.py --tier l --embedders sift_vlad,siglip
```

Small tier first, always — it is the same corpus with fewer distractors, so a
tier-`s` cell is a complete usable artifact in minutes and proves the pipeline
before an hours-long tier-`l` run. **Time a real cell and multiply**; do not
quote an ETA from a tier-`s` cell for tier `l` without accounting for
`sift_vlad`'s per-image feature extraction, which dominates and is roughly
linear in page count.

Cells land in `$VTS_PILE/embeddings/docmarks_<tier>__<embedder>.pkl`. Verify
before trusting:

```bash
python embed_corpus.py --verify
```

`siglip2_l` is available and not in the default list; add it when a study needs
the premium-embedder column, not by default.

## After launching

A submission is not a launch. Confirm each job came back with a numeric id and
that output starts appearing, then arm a completion notification rather than
watching:

```bash
ssh grid 'until [ "$(squeue -u $USER -h -n docmarks-build -o %i | wc -l)" -eq 0 ]; do sleep 120; done; echo DONE; ls -la '"$VTS_DOCMARKS_OUT"
```

Poll the real signal — page counts in `build_report.json`, cell sizes in
`embed_corpus.py --list` — not just `squeue`. A drained queue with missing
output means failures, not completion.

## What to check when it finishes

- `build_report.json`: `tier_cumulative` hits the budgets, `warnings` is empty
  or explained, `rejection_reasons` holds nothing surprising.
- Roster drift: a warning naming classes "absent from this build" means the
  roster and the clustering have diverged — the eval would silently shrink.
- `separations_honoured` matches the number of adjudicated pairs.
- `needs_hand_crop`: band-located classes still owe a hand-drawn query crop.
- `embed_corpus.py --verify`: every cell loads, every media has a vector.

## Growing the corpus later

A build over a *different* page set is a new corpus version unless you pin the
tier cutoffs:

```bash
python build_corpus.py ... --pin-tiers $VTS_DOCMARKS_OUT/build_report.json
```

Pinning keeps tier membership stable so the new numbers stay comparable to the
old ones, at the cost of letting the page counts drift. Without it, say plainly
that it is a new version and re-run the baselines.
