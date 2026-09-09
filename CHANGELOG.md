# Changelog: `vtsearch`

User-facing changes to the `vtsearch` Flask + Angular application. The
companion `vtscore` library has its own CHANGELOG at
[`vtscore/CHANGELOG.md`](vtscore/CHANGELOG.md).

`vtsearch` is **not** versioned with traditional semver. The `__version__`
attribute is the UTC timestamp of `HEAD`'s commit (ISO 8601, Z-terminated),
computed from git at import time in `vtsearch/__init__.py`. Every commit on
`dev` is effectively a new release; there is no tracked version constant to
bump and no per-release tag.

This CHANGELOG is therefore a **curated** record of notable changes and does
not list every commit. Use `git log` for the full history.

## Unreleased

### Changed

- **Opening a saved dataset paces its progress bar for the branch its coverage
  atlas actually takes.** That step either restores the atlas cached in the
  dataset's pickle (~10 ms) or rebuilds a hierarchical k-means from scratch
  (0.0026 s/item, so seconds at the sizes anybody has swept and minutes only
  near the auto-build threshold) - measured 110-700x apart on the same
  datasets - and a
  timing profile could hold only one number for both, so whichever it held made
  the other case's bar up to 0.94 of a bar wrong. A profile can now carry
  coefficients for each branch, and the load route names the branch it is taking
  as soon as it knows, before the expensive part starts. It also remembers on
  the registry entry which branch that dataset took last time, so the weights
  are right from the first update rather than from the middle of the load.
  Pacing only: nothing about what is loaded or stored changes.

### Fixed

- **Startup no longer stalls for minutes on a cold NFS install** (issue #3715).
  `transformers` builds `importlib.metadata.packages_distributions()` when it is
  imported, and the stdlib version of that stats every file recorded by every
  installed distribution - ~85,000 of them in a venv carrying torch, onnx and
  RAPIDS. With the venv's metadata in page cache that is milliseconds; on an
  NFS-mounted venv whose dentries have been evicted it is 85,000 serialized
  round trips, measured at 78/sec on the GRID: **16 minutes 23 seconds** during
  which the process sat in `D` state after printing
  `Running in PRODUCTION mode` and looked hung. Startup now installs a
  stat-free replacement that reads each distribution's `top_level.txt` (falling
  back to parsing `RECORD` as text) before anything can import transformers, so
  the walk never happens.

- **A mixed-media dataset is no longer scored against another embedding space's
  vectors.** Asking the matrix layer for a specific embedder checked only the
  *first* media to decide whether that name was the dataset's primary - and if
  it was, every media then contributed whatever vector it happened to carry.
  On a dataset holding, say, native images in one space and videos in another,
  that stacked the videos' vectors into the images' matrix and scored them
  through the images' detector head: no error, plausible-looking numbers, and a
  different answer if the same items were loaded in a different order. It was
  reachable whenever the two spaces share a dimension, which is common. The
  check now requires every media to agree, so a request for one space reads
  only that space and reports the media it had to leave out.

- **A CLI Auto-Find run that converts or re-clips its dataset now calibrates the
  threshold on what it actually scores.** The cut was fitted on the loaded
  medias while scoring read the converted frames, pages or clips those medias
  fan out into - and those are systematically different populations, because a
  media's score is the *max* over its sub-items and is therefore never below its
  own whole-item score. A cut chosen as a quantile of the first distribution
  lands far lower in the second, so the run reported more hits than the
  algorithm ever chose, with nothing in the output looking wrong. On a dataset
  the detector needs no conversion or re-clip for - the common case, and the
  only one anyone had noticed - the two populations are the same set and
  thresholds are unchanged. Everywhere else the threshold moves, generally up.
  A pure converter route was the sharper edge of the same bug: none of the
  loaded medias carried a vector in the detector's space yet, so the population
  estimator saw an empty haystack and silently did not run at all, shipping the
  plain cross-calibration cut. Both now read the routed snapshot, which is
  prepared once and shared between calibration and scoring rather than built
  twice.

- **The Dashboard's `⋯` › Export labels now exports the detector whose row you
  clicked.** It exported the *active* pair's live labels instead - whatever the
  top-bar pulldown was on, which the row's checkbox does not change - so the
  same row gave three different answers depending only on where the app had
  been: the right labels when the pulldown happened to agree, nothing at all
  after a refresh left no detector active, and the entire dataset after a Find
  run, because Find fills that pair's votes with the detector's own call for
  every item (they are presumptions, deliberately kept out of the labelset, and
  the export was reading them as labels). The row action now names its detector
  and reads that detector's persisted labelset - the exact artefact that
  re-imports as the detector - so it is right whatever else the app is pointed
  at. The Find and Train windows' exports are unchanged: there the live session
  *is* what you are exporting.

- **The text-sort progress bar no longer reserves most of itself for a model
  load that has already happened.** `text_sort` paces three steps -- load the
  embedder, embed the query, score every media -- and the first one is either
  seconds (the first sort a server process runs) or *exactly zero* (every sort
  after it, since the encoder stays resident). One static weight vector cannot
  serve both, and every vector the app has ever shipped or measured was the
  cold one: [#3521](docs/experiments/2026-09-03-drive-cold-3521/REPORT.md)
  scored both fitted timing profiles and the shipped defaults at 0.80-0.85
  *bar error* on this task -- the fraction of the bar budgeted to the wrong
  step -- while their per-step predictions were off by only ~20 %. So the most
  frequently run bar in the app spent three quarters of itself parked on a step
  that was already over. The sort route now asks whether the encoder is
  resident (it is a lookup, not a guess) and drops the load step out of the
  pacing when it is, which is the overwhelmingly common case.

- **CLI autodetect no longer drops media the GUI keeps, and its detector
  threshold no longer moves as a result.** The CLI forced *thin* loading on
  every import -- storing a path reference in place of each media's bytes --
  while the GUI passed the user's "Reference files in place" choice. Thin is
  only a deferral when something outside the source can reproduce the bytes; a
  self-contained pickle, or an importer whose items have no local file, lost
  its only copy. Those media could not be embedded, so they were skipped at
  scoring, and because a detector's threshold is calibrated against the
  population actually being scored, the smaller survivor set also moved the
  cut. The same dataset and detector therefore returned more hits, at a lower
  threshold, from the CLI than from the GUI. Two fixes: the pickle loader now
  keeps inline bytes that nothing can re-read (thin still drops bytes that are
  also on disk, in an archive member, or behind a URL), and the CLI honours the
  importer's own `--reference-files` / `--no-reference-files` flag -- which
  previously did nothing at all -- defaulting to off, as the GUI does.

- **Staging a dataset for the combine flow now queues behind the same
  concurrency limits a regular import does, and a cancelled staging no longer
  reports itself as a failure.** Staging ran its importer and its embed pass on
  a bare background thread that acquired neither the download nor the embed
  gate, so N stagings fetched and embedded fully in parallel with each other
  *and* with gated imports -- exactly the pressure the
  `max_concurrent_dataset_downloads` / `max_concurrent_dataset_embeddings`
  settings exist to bound. On a RAM-constrained host, staging a handful of
  image datasets at once multiplied resident model weights past the configured
  budget with nothing to stop it. Staging now makes the same download -> embed
  gate handoff as an import, and shows the same "Waiting for other datasets..."
  message while queued. Separately, staging's error handling had no branch for
  a user-requested cancel, so stopping one surfaced the raw exception text and
  the dashboard and toasts read it as a red failure; both import paths now
  share one exception taxonomy and a cancel reads as `Cancelled` either side.

- **Autopilot opens on the detector you already trained, instead of on its text
  hint.** Whether a detector arrives already trained -- and so whether Autopilot
  ranks with the model from its first screen instead of seeding from the text or
  example hint -- was decided the instant the Autopilot panel mounted. On entry
  to the Train window that instant is always two round trips before
  `/api/votes` can answer (the window ends any live Find session before it reads
  the votes), so the labelset counts it read were the not-yet-loaded zeroes and
  the answer was always "untrained", however trained the detector was. Only the
  Manual -> Autopilot tab switch, which rebuilds the panel with the counts
  already loaded, ever got it right. The mount-time reading is now a guess that
  the run's first *real* reading of the labelset corrects, and the ranking
  follows that correction. It matters most for a detector trained on one dataset
  and opened on another, where there are no votes to move the phase and so
  nothing else would ever have corrected it. Two smaller things read the same
  flag and were deciding from a value that was false for reasons of timing
  rather than of fact: the re-sort prompt, and the sort mode Autopilot leaves
  selected when you stop it.

- **Find now calibrates each detector against the corpus it is searching, and
  scores each head the way that head was trained.** Two independent faults met
  in the cross-dataset Find route. A detector that is not loaded against the
  dataset being searched is retrained on the fly, and that retrain was handed no
  haystack -- so its Good/Bad line came from the pooled cross-calibration rule,
  which caps false negatives and only floors false positives, i.e. sits below
  the cut the data supports and returns more matches than the detector actually
  makes (on a synthetic 2k-media patch corpus it called all 2000 a match, where
  the population-fitted cut called 728). The corpus was in memory the whole
  time; it is now what the threshold is fitted on. Separately, both scorers read
  one image-level vector per media, which is right for the on-the-fly head
  (trained on image-level label vectors) but wrong for an already-loaded one,
  whose threshold was calibrated on max-pooled patch scores -- so on a patch
  dataset a loaded detector was compared against a line drawn for a higher
  quantity and under-returned. Each path now scores at the geometry its own head
  was trained and calibrated in. A media with no usable vector in the scored
  space is also reported as `N/A` instead of falling out of both result tables.

- **Switching dataset or detector in the Train window now re-ranks the pair you
  land on.** Opening the window seeds a ranking -- Autopilot activates, its
  seed sort runs, and an item lands in the centre. A pair *switch* re-ran none
  of that: it reloaded the media list and the votes and left the new pair with
  the empty ranking the reset had just installed, so you arrived at an empty
  work queue and a placeholder centre with no way back short of leaving and
  re-entering the window. Two faults had to meet. The reload spends a moment
  with the vote cache cleared, and Autopilot read that moment as "this detector
  has no labels" rather than "the labels have not arrived yet" -- on a dataset
  whose embedder cannot search by text that dropped you into Manual for good,
  and rewrote the sort mode on the way out. And the only re-sort a switch could
  ever produce was an Autopilot *phase change*, which a pair that lands in the
  phase you left it in never fires. Autopilot now waits for the vote read
  before deciding it has nothing to work with, and a switch that nothing else
  ranked falls back to the sort a fresh entry would have run -- learned sort
  when the detector has both label classes, otherwise Autopilot's text or
  example seed. The re-rank never moves the centre off an item you have already
  started labelling.

- **Switching dataset or detector no longer leaves the previous pair's item in
  the centre viewer.** Media ids are per-dataset, so the pair switch cleared the
  ranking, the threshold and the vote cache but left the *selection* pointing at
  an item that only existed under the pair you just left. Because the viewer
  stamps the dataset id into the media URL when it builds it, the stranded item
  kept loading from its own dataset and rendered normally -- so the grid showed
  the new pair while the centre showed the old one, with nothing on screen
  saying so. The selection is now dropped with the rest of the pair's state; the
  centre reads "Select a media item to view" until the new pair's first pick
  lands, exactly as it does on a fresh entry.

- **The "arranging your items" wait now shows a remaining-time estimate.**
  Preparing a subset map from Find results renders an ETA chip beside the
  progress bar, but the projection build's status payload never carried an
  estimate for it to show, so the chip was always blank. Background jobs are
  now backed by the same progress tracker the dataset loads use, which derives
  one from the rate the build is actually sustaining (rebased at each phase
  boundary, published on a coarse sticky ladder so the figure doesn't twitch),
  and the projection status payload passes it through as `eta_seconds`.

### Added

- **New demo dataset: Rico Icons -- screenshots with boxed, labelled icons.**
  Four new image demos (`rico_icons_s/m/l/a`) built on the Rico UI-semantics
  corpus: 66,261 Android screenshots whose annotations sit on the *elements*
  rather than the screen. Each media carries the icon classes visible on it as
  multi-label categories (32 curated semantics -- Search, Back arrow, Overflow
  menu, Notification bell, ...) plus one ground-truth bounding box per icon
  instance. This is the first demo in the tree that boxes something *inside* a
  screenshot: every other born-digital source (Enrico, RICO App UIs) labels the
  screen as a whole, and the only two boxed sources (Visual Genome, OpenLogo)
  are natural photographs. Boxes are stored normalised, exactly like Visual
  Genome's.

  Unlike every other demo, the four size variants advertise **different**
  download figures (~0.9 / 1.1 / 1.5 / 8.3 GB). The corpus's screenshots run to
  ~7.7 GB across 67 shard folders, so the loader fetches the annotation manifest
  first, slices it, and then pulls only the shard folders that slice actually
  lands in -- an (S) load costs two folders, not sixty-seven. Re-loading, or
  moving from (S) to (M), pays only for the shards it adds.

### Removed

- **Four REST endpoints with no consumer are gone.** `GET /api/dashboard/dataset-info`,
  `PUT /api/dashboard/dataset-rename`, `POST /api/dataset/load-folder` and
  `POST /api/votes/seed-from-examples` had no caller anywhere in the app -- the
  SPA reads dataset metadata and renames through `/api/datasets/registry`,
  loads folders through the generic importer flow, and seeds examples as part
  of loading a detector. Nothing in the UI changes. An out-of-repo script
  calling one of them directly will now get a 404; the same work is available
  through `GET /api/datasets/registry`,
  `PUT /api/datasets/registry/{id}/rename`,
  `POST /api/dataset/import/server_folder`, and detector load respectively.
  (Issue #3438.)

- **Old settings-file shapes are no longer migrated forward.** VTSearch used to
  carry three shims for settings written by older versions: a one-shot rewrite
  that split a pre-tier-split `data/settings.json` across the two files, and a
  pair of coercions that read a pre-enum boolean `show_animations` as
  `"show"` / `"hide"`. `CLAUDE.md`'s backwards-compatibility policy allows
  breaking saved data freely and forbids exactly these shims, so they are gone.
  A value the settings models reject -- one written before a field changed
  shape, or a hand-edit out of range -- is now **ignored on load** and the
  field's default applies. Your file is never rewritten, so nothing is lost:
  fix the value and it takes effect on the next start. Concretely, a boolean
  `show_animations` now reads as `"show"` (the default) rather than mapping
  True-ish to `"show"` and False-ish to `"hide"`, and `PUT /api/settings` now
  rejects the boolean with a 422 instead of silently rewriting it. Per-user
  keys sitting in `data/settings.json` are inert rather than migrated into the
  default user's file. The one deliberate tier exception is unchanged: the
  built-in `default` user still reads `autofind_detectors`,
  `autofind_exporter` and `autofind_exporter_field_values` through to
  `data/settings.json`, which is what keeps the CLI's `--settings` flat-file
  workflow working. (Issue #3413.)

### Changed

- **Your Dashboard selection now survives leaving the Dashboard.** Going to
  Train and back used to drop the highlighted rows and blank the top bar to
  "Select a dataset" -- even though the pair you had just opened was still
  loaded -- because the selection lived on the Dashboard component and died
  with it. It now lives in `DashboardSelectionService`, so the rows come back
  highlighted and the top bar keeps naming them. The detector grid's
  Drafts/AutoRun tab travels with the selection it scopes, so returning can no
  longer leave a hidden AutoRun row feeding the section actions. (Issue #3445.)

- **Switching between two loaded detectors no longer throws away the labeling
  indicators' cached work.** The Smart / Stable per-step cache retrains one MLP
  per label-history step, and it used to be a single slot stamped with whichever
  `(dataset, detector)` pair last touched it -- so re-selecting a detector you
  had already been labeling in dropped the other one's cache outright, and the
  next `/api/labeling-status` poll rebuilt it from step zero. The cache is now
  keyed by the pair, with the three most recent kept warm, so an A-to-B-and-back
  switch costs nothing. The stability pool tensor -- the largest thing the cache
  holds -- is keyed by dataset instead and shared across that dataset's
  detectors, so keeping several pairs warm does not multiply memory. (Issue
  #3390.)

- **"Enrich descriptions" is now a per-model choice, and can no longer make
  your search worse.** The setting averages a typed query over several
  phrasings ("the sound of a dog", "a recording of a dog", "a dog", ...)
  instead of embedding it as typed. Measured across 22 evaluation collections
  and 560 queries, that helps on some models and hurts on others: it is a gain
  on CLAP General (audio) and X-CLIP (video), and a loss on SigLIP (images),
  E5 and BGE (text) and the faster CLAP -- for text it ranked *worse on all 45
  categories tested*. The phrasings are now attached to the models they
  actually help, so turning the setting on is simply a no-op for the rest,
  rather than a small silent cost. The default stays **off**. (Issue #3341,
  following #3127.)

- **The Settings -> Sorting "Enrich descriptions" tooltip described a
  different feature.** It read "Prepend item filenames to text-sort queries to
  improve matching for named items"; the setting has never touched filenames.
  It averages the text-sort query over the embedder's phrasing templates
  ("a photo of ...", "the sound of ...") instead of embedding it as typed. The
  tooltip now says that, and says where it helps: the #3127 measurement across
  every media-type default found it worth +0.014 AP on CLAP audio search,
  inert on images, and -0.057 on text search -- so the default stays **off**.
  (Issue #3127.)

- **The calibrated cutoff no longer counts your own votes in its picture of
  the collection.** The threshold estimator reads score distributions over
  the whole collection to place the Good/Bad line; the voted items' scores
  in those distributions are optimistically shifted (the models were trained
  on them), so they are now dropped before the line is placed. On large
  collections nothing visibly changes; on small ones (demo-sized, or heavy
  voting) the cutoff gets slightly more accurate. The correction steps
  aside when almost everything has been voted — a tiny leftover pool is a
  worse guide than the full collection — so no regime pays for it.
  (Issue #3308.)

- **The Tuning-fraction default is now per-model.** With no explicit setting,
  the Train/Calibrate split of each calibration fold is 0.3 (70% Train / 30%
  Calibrate) for detectors that learn in a single-vector space and stays 0.5
  for patch-grid models — the #3287 measurement found more Train buys
  −0.012 to −0.013 cost on single-vector embedders in every vote band, while
  patch embedders want the incumbent 0.5 in both their voting styles. The
  Settings → Sorting "Tuning fraction" field now reads empty ("auto") by
  default; typing a value pins it for every detector as before, and clearing
  the field returns to the automatic per-model default. Stored settings that
  already carry an explicit `calibration_fraction` keep winning unchanged.
  (Issues #3287/#3290.)

- **The calibration deck is a talk about ideas again, and every reveal has a
  name.** `slides/decks/hold-the-line.deck` was rebuilt end to end: the
  measurement slides are parked after the Questions slide as backup, so the
  main line is what we found rather than how we ran the study; the two slides
  that argued the threshold's double duty in the abstract are gone, and the
  point is now made on the field of items where it can be seen — one detector
  curve, then the same curve cut looser and tighter, then the item the loose
  and tight cuts disagree about, which is also the item the user gets asked. A
  new slide sets up what the Inclusion knob is *for* before the section that
  shows it failing, and the epilogue's two compressed bullets became two
  slides: why the midpoint of two means is not the Bayes-optimal cut, and why
  a region-voted score is an extreme-value statistic. Every page of a build now
  carries a letter after the shared page number (5a, 5b, 5c), the speaker view
  shows the whole build as a lettered contact sheet under the slide, and
  presenter notes that would have been clipped spill onto a continuation page
  instead.

- **The Add Dataset dialog has one consistent vertical rhythm, and Advanced
  really is hidden.** The Folder, Manifest and Demo forms all laid their fields
  out differently: a field's own controls could sit further apart than two
  unrelated fields, the "Folder to import" path and its **Browse** button were
  separated as if they were different questions while the checkboxes below ran
  together with no gap at all, and the folder browser opened as loose rows with
  no frame. Every field in every importer now shares the same spacing, the path
  input and **Browse** sit on one line (matching the Manifest importer's file
  field), and the browser opens in a framed panel under it. Separately, the
  **Advanced** section now shows *nothing* until you open it: **Embedder** and
  the Demo importer's **Convert to** used to appear on the collapsed form
  whenever their value differed from the default — which, on a demo dataset
  that picks its own embedder, was most of the time. Any non-default choice is
  still disclosed, in the **Advanced** toggle's tooltip. (#3215)

- **Autopilot's "you're done" hand-off is a dialog you answer, and it appears
  once.** Reaching the end of training used to raise a toast that counted down
  and then returned you to the Dashboard unless you cancelled it — and because
  the countdown re-armed on every entry to the Train window, anyone who thought
  their detector needed more work had to dismiss the same redirect each time
  they came back. It is now a **Detector Trained** dialog with two plain
  buttons, **Continue Training** and **Head to Dashboard**, and nothing happens
  until you pick one. It is also raised only for the autopilot run that
  actually trained the detector: continuing afterwards, or picking up a
  detector already trained on another dataset, no longer announces anything.
  (#3201)

- **The detector head is now a linear SVM.** Every trained detector — new
  detectors, saved ones re-derived from their labels, and the per-step models
  behind the labeling-progress indicators — is fitted to the class-balanced
  maximum-margin boundary between your Good and Bad votes (scikit-learn's
  `LinearSVC`) instead of by logistic regression. It is the same shape of model
  as before, a single linear boundary over the embedding space, so nothing
  about detector files, exports, or the ONNX bundle changes: only where the
  boundary lands. Measurements in a separate environment put the SVM's ranking
  clearly ahead of the logistic head's, and while *why* is still under
  investigation, the best-measured head is the one that ships. Detector scores
  will move — a detector retrained after this change can put items in a
  different order and cut at a different threshold than the same votes did
  before. The regularisation strength is tunable via `VTSEARCH_SVM_HEAD_C`
  (default `1.0`), and `VTSEARCH_TRAIN_EPOCHS` / `VTSEARCH_TRAIN_PATIENCE` no
  longer affect a detector fit. See [`docs/ML.md`](docs/ML.md).

### Fixed

- **A results exporter's `dynamic_options` select is now filled in, in both
  places an exporter is configured.** A `"select"` field declared with
  `dynamic_options=True` is supposed to have its options computed at runtime by
  the plugin's `get_field_options()`, which is how an exporter whose
  destinations are only knowable then (mailboxes, buckets, remote queues) offers
  real choices. Exporters never had the route the importer families have, so the
  dropdown rendered the (usually empty) list frozen into the plugin definition —
  the only way to get options was to hard-code them at definition time. Both the
  Export modal and Settings › Auto-Find's **Results Exporter** tab now fetch the
  live list (`POST /api/exporters/field-options/<name>`), re-fetch a field when
  something it declares in `depends_on` changes, show fetch failures inline, and
  honour `allow_free_text` by rendering a combobox. (Issue #3360.)

- **One file a host won't serve no longer sinks a whole multi-file demo
  download.** The Apollo 11 Mission Audio demo pulls its tracks as separate
  files from the Internet Archive, which serves each one from a data node that
  can answer HTTP 500 for minutes at a time while its siblings stay healthy.
  A single such track spent its six retries, then failed the entire dataset
  load with a raw `500 Server Error` naming a `dn721903.ca.archive.org` URL the
  user had never asked for. A file the host refuses is now set aside, retried
  once after the rest of the set (by which point a wobbling node has usually
  recovered), and otherwise skipped with a note in the progress line — the
  download only fails when more than a quarter of the set is missing, and a
  skipped track is fetched the next time the dataset loads. When a server does
  keep returning 500/502/503/504/429 until the retries run out, that now reads
  as *"archive.org kept returning an internal server error (HTTP 500) on all 6
  attempts. That is a problem on the server's side, not yours…"* rather than a
  raw `HTTPError` ending in a CDN node URL. (#3227)

- **A demo download that can't reach its host now says so in a sentence, and
  tries harder before giving up.** Loading the Apollo 11 Mission Audio demo
  while archive.org was refusing connections failed with a hundred-line
  `MaxRetryError` traceback pasted into the Dataset-load-failed box, after
  six identical 10-second connection attempts — and the progress bar said
  "resuming" a file that had never downloaded a single byte. The failure now
  reads *"Couldn't reach archive.org: the connection timed out before the
  server answered, on all 6 attempts. The site may be down or blocked by your
  network/proxy…"*, the connect budget escalates across retries (10 s → 30 s)
  so a host that is merely slow to accept isn't written off six times over,
  the retry notice says "retrying" until there are bytes to resume, and the
  track-list fetch that precedes the download gets the same retry budget
  instead of dying on a single unlucky handshake. (#3216)

- **CLI autodetect no longer reports every media as a hit, and no longer
  disagrees with the GUI's Find.** A run against a dataset and detector that
  the GUI cut at ~0.475 came back with a threshold of `-0.375` and a positive
  hit for every image. Two separate defects were behind it. First, a media the
  detector head cannot score (a corrupt vector, a destabilised fit) is recorded
  at the `-1.0` sentinel — deliberately outside the sigmoid range so it can
  never clear a threshold — and those sentinels were being fed to the threshold
  estimators as if they were scores. A spike a full unit below the range pulls
  the fitted cut under zero, at which point every real score clears it: 2%
  unscorable media moved a threshold from `+0.58` to `-0.50` and a 30-of-600
  shortlist to 588 of 600. Every estimator now fits on the scorable population
  only — fold haystacks, held-out anchors, the pooled conformal orderings, and
  the blend's GMM — and the run warns, naming the count, when media had to be
  excluded. Second, CLI scoring forwarded each media's image-level vector while
  the threshold it compared against (and the GUI) max-pool the media's patch
  rows; on a patch dataset that is a different distribution, enough to turn a
  GUI Find's dozens of hits into zero CLI hits. Both now build their rows
  through the same builder. (#3180)

- **One image that fails to embed no longer aborts a CLI run.** Every CLI
  scoring path fed the raw media chunk straight to the embedding-matrix
  builders, which raise on a media with no vector (`ValueError`, from
  `vtscore/embedding/matrix.py`) or one whose vector is the wrong width
  (`MismatchedVectorError`, from `vtscore/embedding/precomputed.py`). Those
  raises are correct for the dataset's own matrix — the load pipeline has
  already dropped vector-less media by then — but not for a snapshot handed to
  the scorer, so a single corrupt or undecodable file took the whole run down
  with it. The scoring paths now filter first and score what is left, matching
  the drop-and-log policy the load pipeline and converter routing already used,
  and each skip is announced on the CLI event stream (`medias_skipped`) so a
  short hit count is never silent. This also fixes
  `--autodetect --importer …`, which failed on *every* run: the safe-threshold
  population pass scored the chunk before anything had been embedded, so it hit
  the same raise on the first media. (#3179)
- **Rotated photos are no longer processed sideways.** A JPEG carrying an EXIF
  orientation tag — the ordinary output of a phone camera, which stores the
  sensor frame plus "rotate me" rather than rotating pixels — reached the
  embedder, OCR, face detection and every crop path 90/180/270 degrees from the
  way the browser (and every other photo viewer) displays it. Nothing errored;
  the picture was simply the wrong way up in a space nobody looks at directly,
  so a sideways photo embedded as a sideways photo and landed in the wrong part
  of the vector space. Orientation is now applied once, in the image decode
  layer, so every consumer sees the same upright pixels; a media's stored
  `width`/`height` are the displayed dimensions to match, and cropping, lazy
  clip resolution and cross-dataset origin resolution decode upright so a box
  drawn on screen cuts the region the user drew. The `image_exif_orient`
  cleaner still exists for anyone who wants the rotation baked into the *stored
  bytes* (for tools outside VTSearch that ignore EXIF), but it is now off by
  default: it costs a lossy re-encode and no longer changes anything VTSearch
  itself sees. Images already imported keep the dimensions and embeddings they
  were ingested with; re-import a rotated corpus to pick up the fix. (#3172)

- **A detector's labelset no longer stores the same media twice.** After one
  pass over a 300-image dataset a detector reported `num_training: 356` — 300
  distinct images, 56 of them held as a duplicate pair. The two halves of the
  labelset round-trip disagreed on what "the same media" means: loading a
  detector turned an entry into a vote when it matched a dataset item by
  origin **or** content hash, while saving decided an entry belonged to the
  active dataset by comparing origins alone. Anything that matched only on its
  hash — an exemplar carrying the `example_media` sentinel, a label imported
  from a plain md5 list, a label saved under another dataset's importer — was
  restored as a vote, re-emitted as a fresh element, and kept beside its
  original. Both writers now use the same origin-or-hash resolution, so a
  re-vote updates the existing element instead of appending, and a labelset
  that already carries duplicates collapses on the next write. Two related
  leaks went with it: the Find "add corrections to detector" fold left the
  stale entry in place beside its correction (one media, two contradicting
  labels), and a drawn Good region was erased whenever a detector reload
  resynced the labelset, because the restored vote came back image-level.
  Duplicated entries were also double-weighted at training time. (#3174)

- **A finished dataset import no longer looks like a wedged one.** An import
  that had already succeeded — pickle written, dataset registered — kept the
  progress channel parked on its last message ("Loading SigLIP processor…")
  indefinitely, with no loader thread left in the process. Three things fed
  that: a model load taken through the app-wide progress sink narrated itself
  on the dataset channel and never said it had finished, the load task's
  success path wrote no terminal state (only its failure paths did), and
  `POST /api/dataset/cancel` answered `{"ok": true}` while doing nothing,
  because cancellation is cooperative and there was no worker left to observe
  the flag. Model loads now terminate whatever channel they borrowed
  (background warm-ups are silent, and an import's model load reports on the
  import's own row); a load parks its tracker when the work ends, whichever
  way it ended; and cancel reports what it actually reached — `409` with
  `ok: false` when it reached nothing, clearing the stale progress on the way
  out. The dataset registry also re-reads its manifest when the file on disk
  changes, so a dataset registered by another process (a CLI `--autodetect`
  run against the same data dir) is listed without a restart. (#3167)

- **Detector load can no longer hang forever at "Preparing".** Loading a
  detector while the selected dataset was not yet (re)loaded — typical right
  after an app restart, while the dataset was still being read from its
  pickle — left the dashboard row stuck at "Loading detector · Step 1 of 3 ·
  Preparing" indefinitely: Cancel did nothing, every retry reported "Detector
  load already in progress", voting returned 409, and only a restart
  recovered. The load now fails fast with a clean 409 (`dataset_not_loaded`)
  in that case, any other failure before the background worker starts
  releases the load reservation and surfaces an error on the row, and load
  failures are written to the server log. (#3139)

### Changed

- **A seed importer's results now appear on its own tab.** Running a seed
  importer in the New Detector modal used to switch the user to the media
  tab, where the batch had been appended — an odd jump mid-import, and one
  that hid the form they were still working in. The example stack is now
  mirrored under each seed importer's form, so seeds land in view where they
  were added. It stays one list: the mirrored rows carry the same Seed badges
  and Remove buttons, and edits from either tab hit the same stack. Picking an
  exemplar by hand still lands on the media tab, which is where the picker
  lives. (#3192)

- **Image preprocessing now names its backend instead of inheriting one.**
  Every image embedder builds its processor by asking `transformers` for the
  `torchvision` backend outright (`VTSEARCH_IMAGE_PROCESSOR_BACKEND`, new
  default `torchvision`; set `auto` for the previous behaviour). Nothing in the
  code used to say which implementation resized and normalised an image, and
  the answer changed *inside* the version range we pin: `transformers` 5
  removed the `Fast` suffix, so the bare `SiglipImageProcessor` means the PIL
  implementation below 5 and the torchvision one at 5+, while
  `requirements/image-embedders.txt` asks only for `>=4.49`. The two are not
  interchangeable — they disagree on 53–59% of pixel elements and by a median
  `1 − cos` of ~1.5e-04 on `siglip2_l`, 50× the perturbation half precision
  causes — so two hosts resolving different wheels produced different vectors
  from identical code and weights, with nothing recording which. **On a
  `transformers` 5 host this changes nothing** (the pre-embedded pile is
  torchvision-built, reproduced to 7.6e-13). **On a 4.x host it changes the
  vectors**, which is the point: that host was quietly disagreeing with the
  pile and now agrees with it. Because a backend request is a request and not a
  guarantee — DINOv3 ships no PIL implementation, and `transformers` warns and
  falls back rather than raising — each embedder now reads back the class it
  actually loaded and logs a warning naming itself when it differs. (#3173)

- **Bad pre-computed vectors are now rejected at import, with an error that says
  what is wrong.** Importing an `.npz` manifest of pre-computed embeddings used
  to accept anything: vectors of the wrong width for the embedder the manifest
  named, `NaN`/infinite rows from a failed embed, ragged archives, `float64` or
  half-precision exports. None of those failed at import. A wrong-width row
  surfaced later as `could not broadcast input array from shape (768,) into
  shape (1152,)` on an unrelated search, naming neither the file nor the
  manifest; a non-finite row never raised at all and silently corrupted every
  score and threshold it touched. Manifests are now checked as they are read —
  including against the declared embedder's own dimension, so an archive that
  says `siglip2_l` while shipping 768-dim rows is caught immediately — and
  vectors are widened to `float32`, so a half-precision or double-precision
  export imports cleanly instead of leaving the dataset mixed. If a dataset
  still ends up holding two widths, sorting and training now name the offending
  item and both dimensions instead of failing with a bare numpy shape error.
- **Audio now defaults to the larger CLAP checkpoint.** New audio datasets and
  text queries use `clap_general` (`laion/larger_clap_general`, shown as "CLAP
  (general, larger)") instead of `clap`. It wins every measured retrieval
  comparison on ESC-50, at roughly 2.1x the embedding time. The old checkpoint
  is still selectable as "CLAP (general, faster)" for large collections where
  ingest speed matters more, and existing datasets and detectors built with it
  keep working. Cached demo-dataset pickles built with `clap` are re-embedded
  the next time they are loaded with the new default.
- **Library extracted.** The reusable core of VTSearch was carved out into a
  separate `vtscore/` package. The user-facing application surface (the Flask
  app, the Angular SPA, the settings system, the auth layer) is unchanged.
  Internally, every library-candidate import path moved from `vtsearch.<lib>`
  to `vtscore.<lib>`; `vtsearch/state/__init__.py` is now a thin app-tier shim
  that re-exports `vtscore.state` and layers the proxy view (`medias`,
  `good_votes`, …) on top. See
  [`vtscore/docs/architecture.md`](vtscore/docs/architecture.md) for the
  seven seams the refactor introduced.
- **Plugin entry-point groups renamed.** Library-tier plugin families now
  register under `vtscore.<family>` instead of `vtsearch.<family>`
  (`vtscore.importers`, `vtscore.label_importers`, `vtscore.labelset_sources`,
  `vtscore.media_sources`, `vtscore.converters`). Settings-related families
  remain under `vtsearch.<family>` because they stay app-side
  (`vtsearch.settings_importers`, `vtsearch.settings_exporters`,
  `vtsearch.settings_sources`). Third-party plugin authors targeting the
  library tier need to update their `pyproject.toml` entry-point group
  names.

### Added

- **Three long-form audio demos: Apollo 11, BirdVox Full Night, and the Nixon
  White House Tapes.** Every audio demo so far was a corpus of short labelled
  clips (ESC-50, GTZAN, UrbanSound8K) or, in TUT's case, 32 four-minute street
  soundscapes. These three are hours-long *unlabelled* recordings where the
  interesting content is discrete events scattered through the runtime — the
  Quindar beeps, master alarms and MOCR applause in 174 hours of NASA mission
  loops; the sub-second bird flight calls in six ten-hour night recordings from
  BirdVox-full-night; the telephone rings, laughter and room noise under 12
  tapes' worth of Nixon's secret taping system. Each loads as one
  undifferentiated bucket, so you clip it yourself, vote on a handful of hits,
  and let the detector rank the rest. All three sources are freely
  redistributable (CC PD Mark, Creative Commons, and US federal public domain
  respectively).

  Unlike the older demos, **each size variant downloads only its own slice** of
  the source rather than the whole thing — at 5-10 GB apiece that difference
  matters, so (S) costs a twelfth (Apollo, Nixon) or a sixth (BirdVox) of the
  figure shown in [`docs/demos.md`](docs/demos.md). BirdVox's ten-hour FLAC
  units are segmented into 10-minute chunks as they download, since a ten-hour
  file cannot be handed to the clipper as a single item.

- **Seed importers: a new plugin family for unlabeled seed media.** An
  external package can now contribute its own tab to the New Detector modal's
  **Blank** flow, beside Text and the media picker, by registering a
  `SeedImporter` in the `vtscore.seed_importers` entry-point group. Where a
  label importer imports media that already carry a good/bad verdict, a seed
  importer imports a *batch* of media with **no verdict** — items that are
  "close but not quite" what the user is hunting for. Seeds are stored on the
  detector as `{"type": "media", "value": …, "labeled": false}`: they steer
  the first sort (Autopilot ranks against the centroid of every media
  example) but never become a Good label or vote, so a detector seeded this
  way starts untrained. Nothing ships in-tree, so an install with no such
  plugin looks exactly as before. New endpoints: `GET /api/seed-importers`,
  `POST /api/seed-import/<name>`, `POST /api/seed-import/<name>/options`.
- **Server-side code can raise a toast.** A new `notify()`
  (`vtscore/concurrency/notifications.py`) lets any backend code — most
  usefully a plugin that hit a recoverable problem — tell the user something
  happened *without* failing the operation: "skipped 3 unreadable files",
  "the remote API rate-limited us, results are partial". The message is
  broadcast over a new `notification` channel on `/api/events` and rendered
  as a toast; toasts gained `warning` and `info` levels alongside the
  existing `error` and `success`. Plugin subclasses get `self.notify(...)`
  with their display name attached. Headless runs print the same messages
  (stderr in text mode, `notification` NDJSON records under
  `--progress-format json`). Delivery is live-only — there is no replay for
  a client that connects afterwards. See
  [`docs/EXTENDING-plugins.md`](docs/EXTENDING-plugins.md#notifying-the-user-toasts).

- **The app now tells you when your browser is running an out-of-date build.**
  `static/` is a build artifact that git does not track, so pulling new code and
  restarting the server used to leave the browser loading whichever bundle was
  last built — silently, since the version in Settings is the *server's* and
  looks current regardless. The bundle now carries the commit it was built from,
  and a mismatch raises a toast naming both versions and the rebuild command,
  plus a `⚠ bundle v …` chip beside the version in the Settings footer.

- `vtscore` library distribution with its own [README](vtscore/README.md) and
  [CHANGELOG](vtscore/CHANGELOG.md). See the
  [package reference](vtscore/docs/README.md#package-reference) for the
  documented public surface.
