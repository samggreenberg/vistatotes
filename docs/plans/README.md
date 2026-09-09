# `docs/plans/` — index

Plan files describe **work still owed**: a proposed feature, or the open parts of
one in progress. They are *not* an archive of completed work — git history and
merged PRs are the record of what already landed. See `CLAUDE.md` for the full
policy (ship-and-prune, `<!-- item-sep -->` sentinels, issues-vs-plans, and the
rule that a plan may *reference* an issue but must never duplicate its body).

**Where the other archive lives.** Finished measurements are written up in
[`docs/experiments/`](../experiments/) — one directory per study, holding its
`REPORT.md`, the generated tables and figures, and (for a long study) a
self-contained `report.html` reading copy. See [its
index](../experiments/README.md). A plan links *into* those; it does not restate
them.

This index groups plans by area and deliberately says nothing about their
contents: what's owed lives in each plan file alone, so there is no summary here
to go stale. Keep the grouping in sync when you add or delete a plan — one line.

## Frontend

- [`angular-22-upgrade.md`](angular-22-upgrade.md)
- [`httpresource-migration.md`](httpresource-migration.md)
- [`user-docs-screenshots.md`](user-docs-screenshots.md)

## Browse / projection

- [`vtsbrowse.md`](vtsbrowse.md)
- [`vtsbrowse-empirical-tuning.md`](vtsbrowse-empirical-tuning.md)
- [`vtsbrowse-toponymy.md`](vtsbrowse-toponymy.md)

## Embedders, detectors, media

- [`patch-embedder.md`](patch-embedder.md)
- [`structural-embedder.md`](structural-embedder.md)
- [`half-media-types.md`](half-media-types.md)
- [`media-cleaners.md`](media-cleaners.md)
- [`visual-genome-dataset.md`](visual-genome-dataset.md)

## Thresholds and calibration

- [`population-anchored-calibration.md`](population-anchored-calibration.md)
- [`inclusion-calibration-bias.md`](inclusion-calibration-bias.md)
- [`provenance-partitioned-calibration.md`](provenance-partitioned-calibration.md)
- [`calibration-experiment.md`](calibration-experiment.md)
- [`threshold-stability-experiment.md`](threshold-stability-experiment.md)
- [`region-vs-binary-kappa-mechanism.md`](region-vs-binary-kappa-mechanism.md)

## Scoring and eval

- [`max-patch-experiment.md`](max-patch-experiment.md)
- [`set-scorer-experiment.md`](set-scorer-experiment.md)
- [`coverage-atlas.md`](coverage-atlas.md)
- [`vg-scale-bands-and-corrections.md`](vg-scale-bands-and-corrections.md)
- [`vg-scale-exhaustive-annotation.md`](vg-scale-exhaustive-annotation.md)
- [`stopping-rules-in-eval.md`](stopping-rules-in-eval.md)

## Platform / CLI

- [`scalability.md`](scalability.md)
- [`cli-stream-massive-images.md`](cli-stream-massive-images.md)
- [`cli-detector-converter.md`](cli-detector-converter.md)

## Plugins and I/O

- [`exporter-payload-contract.md`](exporter-payload-contract.md)

## Cross-cutting audits

- [`codebase-audit-2026-08.md`](codebase-audit-2026-08.md)
- [`documentation-accuracy.md`](documentation-accuracy.md)
