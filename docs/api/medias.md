# Medias & Sorting

[← Back to API index](../API.md)

> Media, vote, and sort endpoints are scoped to the active dataset/detector via
> the [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).
> Vote- and label-mutating routes **require** them (400 otherwise).

---

## Medias

### List media IDs

```
GET /api/medias/ids
```

→ Lightweight JSON array of stubs, one per media in the loaded dataset:

```json
[
  { "id": 0, "media_type": "audio", "embedder": "clap-fused" },
  { "id": 1, "media_type": "audio", "embedder": "clap-fused" }
]
```

Every stub carries `id` and `media_type`; `embedder` (and the plural
`embedders` array, when a media was embedded by more than one embedder, e.g.
a semantic + region-patch pair) is included when present.  Display-worthy
metadata (`filename`, `md5`, `custom_metadata`,
`origin_name`, `description`, `clip_*`) is fetched on demand for the IDs
the client actually needs via [Batch fetch](#batch-fetch-metadata); this
keeps the listing payload bounded even for datasets with tens of
thousands of items.

### Batch fetch metadata

```
POST /api/medias/batch
Content-Type: application/json

{ "ids": [0, 1, 2] }
```

→ JSON array of full metadata objects for the requested IDs (unknown IDs
are silently omitted):

```json
[
  {
    "id": 0,
    "media_type": "audio",
    "filename": "media_0.wav",
    "md5": "abc123...",
    "custom_metadata": {
      "duration": 5.0,
      "file_size": 160044,
      "category": "sine",
      "frequency": 440
    },
    "origin_name": "media_0.wav",
    "description": "A 440 Hz sine wave"
  }
]
```

Every returned item contains `id`, `media_type`, `filename`, `md5`, and
`custom_metadata`.  `origin_name`, `description`, `embedder`, `embedders`
(plural array, when present), `has_original`, and `clip_*` keys are included
when present.

`has_original: true` marks an item a
[MediaCleaner](../EXTENDING-media.md#adding-a-media-cleaner) rewrote at load
time, whose pre-clean payload was kept alongside the canonical (cleaned) one.
Those items accept `?variant=original` on every payload route below, and the
detail viewer offers a Clean/Original toggle. The key is absent otherwise.

The `custom_metadata` dict is the media type's display fields — e.g.
`duration`/`frequency` for audio, `width`/`height` for images, `word_count`
for text — with any importer-supplied `custom_metadata` layered on top.

It also carries up to three curated **provenance** lines distilled from the
media's `origin.params`:

| Field | Present on | Example |
|-------|-----------|---------|
| `Source` | Converter / clipper output | `/data/videos/movie.mp4` |
| `Derived Via` | Converter / clipper output | `Video → Images (n_clips=2)` |
| `Imported Via` | Any media whose origin names an importer | `Manifest (paths_file=/data/list.txt)` |

`Source` is the original file the item came from — the video an extracted
frame was cut from, the recording an audio clip was sliced out of.  A plainly
imported file gets neither `Source` nor `Derived Via`; it is its own source.

Each is one line rather than a key-per-`origin.params`-entry, because a
dataset-level import knob (`size=60`) is not a fact about one item and reads
wrong in a per-item grid.  The machine-only replay recipe
(`converter_content_hash`, `converter_out_index`, `clipper_chain`,
`converter_param_*`, …) is folded into these lines rather than listed raw.
The enriched label export (`GET /api/labels/export?enrich=true`) does
flatten the *full* `origin.params` key-by-key — an export is a machine-facing
artifact with opt-in columns, where the raw recipe is the point.

### Payload variants (`?variant=original`)

Every per-media payload route below — `/audio`, `/video`, `/image`,
`/thumbnail`, `/text`, `/paragraph`, `/media` — accepts an optional
`variant` query:

| `variant` | Serves |
|-----------|--------|
| omitted / `""` | The **canonical** payload: the cleaned bytes that were actually hashed, embedded, and scored. |
| `original` | The pre-clean payload of an item a cleaner rewrote at load time. |

`?variant=original` on an item with no snapshot (`has_original` absent) falls
back to the canonical payload rather than 404ing, so a stale link still shows
the item. Any other value is rejected with `422`.

Derived metadata is recomputed from what is actually served rather than reused
from the canonical item: the `original` variant regenerates the thumbnail
(the stored one describes the cleaned bytes), recounts `word_count` /
`character_count` for text, and hashes the served bytes for its `ETag`.

### Stream audio

```
GET /api/medias/{media_id}/audio
```

→ `audio/wav` binary stream.
404 if media not found.

### Stream video

```
GET /api/medias/{media_id}/video
```

→ Video binary stream (`video/mp4`, `video/webm`, or `video/ogg` based on
filename extension). Non-browser-playable formats are transcoded to MP4.
400 if not a video. 404 if not found.

### Stream image

```
GET /api/medias/{media_id}/image
```

→ Image binary stream (`image/jpeg`, `image/png`, `image/gif`, `image/webp`, or
`image/bmp` based on filename extension).
For a non-image media type the route delegates to that type's
`image_response` hook, so audio serves its waveform PNG, video its
mid-frame, and a PDF document its first page.
400 if the media is not an image and its `image_response` hook yielded
nothing. 404 if not found.

### Get text content

```
GET /api/medias/{media_id}/paragraph
GET /api/medias/{media_id}/text
```

Both paths serve the same handler. Returns the text content and statistics
for a text media item.

→ `{"content": "...", "word_count": 150, "character_count": 900}`
400 if not a text media. 404 if not found.

### Generic media endpoint

```
GET /api/medias/{media_id}/media
```

Delegates to the registered media type's handler. Works for all media types.
400 for unsupported type. 404 if not found.

### Vote on a media

```
POST /api/medias/{media_id}/vote
```

**Body:** `{"target": "good"}`, `{"target": "bad"}`, or `{"target": "none"}`

**Absolute-target semantics**, not toggle semantics: `target` is the state the
media should be in *after* the call, so un-voting is an explicit
`{"target": "none"}` rather than a repeated vote in the same direction.

The call is **idempotent**: sending the state the media is already in is a
no-op — it does not append to the label history, does not credit achievements,
and returns the existing click-time. (This is deliberate: with toggle
semantics, two stale-view tabs clicking the same media alternated
ADD/REMOVE on the server and inflated the achievement counters.)

**Optional `region_box`** (`"good"` targets only): a 4-float array
`[x0, y0, x1, y1]` in normalised image coordinates (`0..1`,
pre-rotation) that annotates *which region of the image* the user
is voting good on. Persisted alongside the vote and consumed by
region-aware head training (the trainer pools the box's patch-grid
cells on the fly). The box is dropped when the vote is removed
(`target: "none"`) or switched good → bad; sending one with a
`"bad"` or `"none"` target is rejected.

```json
{"target": "good", "region_box": [0.2, 0.3, 0.55, 0.7]}
```

**Optional `provenance`**: how the item came to be in front of the user.
Recorded per vote, read by nothing — the recording exists because the
surfacing context is *not re-derivable later*: the ranking is client-side
state and the model behind the score is overwritten by the next retrain, so a
vote not annotated at click time is annotated never. It is stored in the
element's labelset `metadata` under `"vt:provenance"` and round-trips through
label export/import.

Six optional fields, four of them independent categorical axes rather than one
fused enum (the bias this recording exists to measure tracks *how the item was
drawn*, not *who was driving*, and the two come apart — a user can pick the
`hard` select mode by hand and get autopilot's exact margin-sampled draw):

| Field | Values | Meaning |
|-------|--------|---------|
| `flow` | `autopilot`, `list_review`, `find_verify`, `labelset_review`, `seed_example`, `import`, `bulk`, `undo`, `unknown` | Which UI flow drove the vote. |
| `phase` | `good`, `bad`, `hard`, `new` | Autopilot phase; ignored unless `flow` is `autopilot`. |
| `select_mode` | `top`, `hard`, `new` | How the item was drawn off the ranking. |
| `sort_kind` | `learned`, `text`, `load` | Which ranking the user was looking at. |
| `rank_at_vote` | integer ≥ 0 | The item's position in that ranking. |
| `score_at_vote` | float | The item's model score when it was surfaced. |

Recorded **only when the call actually changes the vote state**, so an
idempotent re-send from a stale tab cannot overwrite what the original click
recorded. An unrecognised *value* for any of these fields is rejected (422); a
payload carrying nothing beyond `{"flow": "unknown"}` is dropped rather than
stored.

```json
{"target": "good", "provenance": {"flow": "autopilot", "phase": "hard",
                                  "select_mode": "hard", "sort_kind": "learned",
                                  "rank_at_vote": 12, "score_at_vote": 0.44}}
```

→
```json
{"ok": true, "state": "good", "click_time": 17}
```

`state` is the media's vote state after the call (`"good"` / `"bad"` /
`"none"`), so a client can reconcile its optimistic view without a follow-up
`GET /api/votes`. `click_time` is the click-time ordinal assigned to the new
label, and is `null` when the target was `"none"` or when the call was an
idempotent no-op.

Unknown body fields are silently dropped, so a client may attach advisory keys
(e.g. `confidence`, `note`) without failing schema validation.

| Status | Cause |
|--------|-------|
| `400` | `region_box` on a `"bad"` / `"none"` target; `region_box` outside `[0, 1]`, not a 4-element list, or non-numeric. |
| `404` | Media not found. |
| `422` | Missing `target`, a `target` outside `good` / `bad` / `none`, or an unrecognised `provenance` value (marshmallow validation envelope). |
| `500` | The vote was applied in memory but the detector labelset could not be persisted. |

### Bulk vote

```
POST /api/medias/vote-bulk
```

**Body:** `{"ids": [1, 2, 3], "target": "good"}`, plus an optional
`provenance` block (same shape as the per-media vote) applied to every id in
the batch. It defaults to `{"flow": "bulk"}` when omitted.

Applies one absolute vote `target` (`"good"` / `"bad"` / `"none"`) to many
medias in a single request, with the same idempotent semantics as the
per-media vote (including Find-mode verification: a good/bad target marks the
item verified). The detector labelset is persisted once rather than per id.
Bulk votes are image-level (no region boxes). Powers the Browser's "Verified
Good" / "Verified Bad" actions. Unlike a hand-click, a bulk vote does not build
the Marathoner streak.

→ `{"ok": true, "changed": 2, "missing": [3]}` — `changed` counts only ids
whose state actually moved (idempotent re-applies don't count); ids not in the
loaded dataset are reported in `missing`.
400 if no ids supplied; 422 on an unrecognised `provenance` value.

### Thumbnail

```
GET /api/medias/{media_id}/thumbnail
```

**Query (optional):** `region=x0,y0,x1,y1` (normalised fractions in `[0, 1]`)
crops the thumbnail to a sub-region (used so the Good pile shows a
region-voted item's crop rather than the whole frame).

Streams a downscaled thumbnail bounded to a fixed longest-side length, the
same regardless of zoom level (an `ETag` lets the browser reuse it across
scrolls/zoom). Grid and list tiles use this instead of `/image` so a gallery
of high-resolution items doesn't decode every full-size bitmap at once.
400 if the media is not an image and its `image_response` hook yielded
nothing. 404 if
not found or bytes unavailable.

---

## Sorting

### Sort response shape (windowing)

The sorts that rank the whole dataset — [text sort](#text-sort),
[learned sort](#learned-sort), [example sort (upload)](#example-sort-upload)
and [label-file sort](#label-file-sort) — do **not** return a bare
`{results, threshold}` pair. They return a windowed envelope:

```json
{
  "results": [{"id": 0, "similarity": 0.8234}],
  "threshold": 0.5123,
  "acq_threshold": null,
  "sort_token": "9f1c…",
  "total": 250000,
  "above_threshold": 1840,
  "has_more_below": true
}
```

| Field | Meaning |
|-------|---------|
| `results` | The transmitted ranking rows, **descending by score**. May be a *head window* of the full ranking (see below), not the whole thing. |
| `threshold` | The decision line (see [learned sort](#learned-sort) for `threshold` vs `acq_threshold`). |
| `acq_threshold` | The acquisition cut; `null` on sorts with no detector behind them. |
| `sort_token` | Opaque handle for [`GET /api/sort/page`](#sort-page). Also the sort-generation token: a re-sort mints a new one. |
| `total` | Length of the **full** ranking — `>= results.length`. |
| `above_threshold` | Rows at or above `threshold` across the full ranking (not just the window). |
| `has_more_below` | `true` when `results` is a head window and more rows follow. |

**Windowing only engages on large sorts.** Below `SORT_WINDOW_THRESHOLD`
(20 000 rows, `vtscore/state/sort_results_cache.py`) the full ranking is
transmitted unchanged and `has_more_below` is `false`. At or above it the
response carries only the initial window — up to `SORT_WINDOW_HEAD` (500)
above-threshold rows plus `SORT_WINDOW_TAIL` (200) rows just past the
boundary — and the client pages the rest through `/api/sort/page`.

A client that ignores `has_more_below` therefore gets a **silently truncated
ranking** on datasets past 20 k items. Page until `has_more` is `false` (or
until `offset + results.length == total`) when you need the whole order.

The full ranking is held server-side in a process-global LRU cache of the 8
most recent sorts. Nothing is persisted: it holds only the lightweight
`{id, score}` / `{id, similarity}` rows, never embeddings or model weights.

### Sort page

```
GET /api/sort/page?token=<sort_token>&offset=0&limit=200
```

Returns one window of a previously-computed ranking, so a client can scroll
deep into a large sort without receiving the whole list up front.

| Query | Default | Notes |
|-------|---------|-------|
| `token` | *(required)* | The `sort_token` from the sort response. |
| `offset` | `0` | Start index into the full ranking; `>= 0`. |
| `limit` | `200` | Window size, `1`–`2000`. |

→
```json
{
  "results": [{"id": 4021, "score": 0.4412}],
  "offset": 600,
  "limit": 200,
  "total": 250000,
  "threshold": 0.5123,
  "has_more": true
}
```

404 when the token is unknown, has been evicted from the cache, or belongs to
a different dataset than the active `X-Dataset-Id` — in every case the client
should re-run the sort and start from the new token.

### Text sort

```
POST /api/sort
```

**Body:** `{"text": "dog barking"}`

Embeds the text query using the media type's embedding model, then sorts all
medias by cosine similarity. Includes a GMM-based threshold.

→ A [windowed sort response](#sort-response-shape-windowing) whose rows are
`{"id": 0, "similarity": 0.8234}`. `acq_threshold` is `null` (no detector
behind this sort).

When the dataset's embedder is patch-region-aware (e.g.
`dinov3_patch`), each result additionally carries
`"best_region": [x0, y0, x1, y1]`: the normalised box of the
region whose vector matched best against the query, used by the
gallery card to draw a faint outline. Boxes that cover the full
image (the single-vector fallback `[0, 0, 1, 1]`) are suppressed by
the frontend.

Returns HTTP 400 + `{"supports_text": false, ...}` when the dataset's
embedder doesn't support text queries.

### Text sort progress (SSE)

Text-sort progress streams on the `sort` channel of
[`/api/events`](events.md):

```json
{"status": "sorting", "message": "Computing similarities…", "current": 50, "total": 100}
```

Status is `"idle"` or `"sorting"`.

### Learned sort

```
POST /api/learned-sort
```

Trains the detector head on the current good/bad votes and scores all medias. Requires at
least one good and one bad vote.

**Asynchronous by default.** Training is GIL-bound, so the endpoint hands the
work to a background thread and returns immediately:

→ `{"job_id": "…", "status": "running", "current": 0, "total": 1}`

Poll [`GET /api/learned-sort/result`](#learned-sort-result-poll) with that
`job_id` until `status == "done"` to receive the results. A no-op call (votes,
detector, inclusion, and threshold settings unchanged from the most recent
successful run) short-circuits and returns the cached `done` payload directly.

Pass `{"wait": true}` in the body to block until the job finishes and receive
the result inline (used by tests; the frontend leaves it `false`):

→ A [windowed sort response](#sort-response-shape-windowing) whose rows are
`{"id": 0, "score": 0.9234}`.

The `done` payload — whether returned inline (`wait=true`) or via the result
poll — is that same windowed envelope: `results`, `threshold`,
`acq_threshold`, `sort_token`, `total`, `above_threshold`, `has_more_below`.
This is the only sort with a detector behind it, so the only one whose
`acq_threshold` is non-`null`.

`threshold` is the **decision line**: the cutoff shown to the user, what
`above_threshold` counts against, and what Find calls a match. `acq_threshold`
is the **acquisition cut**, and it is a different number — Autopilot's Hard and
New picks read a threshold as a *rank position* rather than a boundary, so they
sample around a cut taken three inclusion steps below the reporting one, which
places it higher in the ranking. Nothing shown to the user reads it. It is
`null` on sorts with no detector behind them (`/api/sort`, `/api/example-sort`,
`/api/label-file-sort`), where a client should fall back to `threshold`. See
[`docs/ML.md`](../ML.md#threshold-calibration) for the mechanism and the
measurement behind the offset.

#### Learned sort result (poll)

```
GET /api/learned-sort/result?job_id=<id>
```

Polls a background learned-sort job.

- Running: `{"job_id": "…", "status": "running", "current": N, "total": M}`
- Done: the [windowed sort response](#sort-response-shape-windowing), plus
  `job_id` and `status`.
- Cancelled: `{"job_id": "…", "status": "cancelled"}`
- Job failed: HTTP 500.
- Unknown `job_id`: HTTP 404.

#### Cancel learned sort

```
POST /api/learned-sort/cancel/<job_id>
```

Sets the cancel flag on the job; the training loop polls it cooperatively.
Returns `{"ok": true}` (HTTP 200) even when the job has already finished — the
contract is "make sure it's no longer running". Unknown `job_id`: HTTP 404.

On patch datasets the head is max-pooled over each image's score-row
stack (the image-level vector plus every raw patch of its
`patch_grid`), and each result carries `"best_region": [x0, y0, x1,
y1]` for the row whose score won - the whole image when the
image-level row wins, otherwise the single winning grid cell.
Region-annotated Good votes (`region_box` on `LabeledElement`) train
on the raw patch nearest the user's box; Bad votes flood the whole
stack (a region-aware asymmetric loss). See [`docs/plans/patch-embedder.md`](../plans/patch-embedder.md)
for the design.

### Example sort (upload)

```
POST /api/example-sort
```

**Form:** `file`: media file to use as the query example.

Embeds the uploaded file and sorts by cosine similarity.

→ A [windowed sort response](#sort-response-shape-windowing) whose rows are
`{"id": 0, "similarity": 0.8234}`.

`best_region` is included per-result on patch-region-aware datasets,
same shape as text sort.

### Example sort (by loaded media id)

```
POST /api/example-sort-by-id
```

**Body:** `{"media_id": 42}` (optionally `{"media_id": 42, "crop_params": {...}}`)

Sorts all medias by similarity to an already-loaded media item. When
`crop_params` is absent the media's existing embedding vector is reused (no
fetch, no re-embed); when set, the media's bytes are materialised, cropped,
and re-embedded before sorting. Powers the right-click "sort by similarity" /
"crop then sort" context-menu actions.

→ `{"results": [...], "threshold": 0.5123}`

The three `example-sort-{by-id,server,origin}` routes are the exception to
the [windowed sort response](#sort-response-shape-windowing): they return the
plain `{results, threshold}` pair with the full ranking and mint no
`sort_token`, so there is nothing to page.

400 if no medias loaded or `media_id` not in the loaded snapshot. 404 if the
media's bytes are unavailable when cropping is requested.

### Example sort (server files)

```
POST /api/example-sort-server
```

**Body:** `{"filenames": ["example.wav"]}` (optionally with `"crop_params"`)

Same as example sort but uses one or more files already on the server in
the user's `example_media/` directory. With multiple filenames the haystack is ranked
against the centroid (mean of the L2-normalised embeddings) of all
examples — this is how Autopilot's Good phase sorts for a detector seeded
with several media examples. `crop_params` describes a single example, so
it is rejected (400) when more than one filename is given.

→ `{"results": [...], "threshold": 0.5123}`

### List server media files

```
GET /api/server-media-files
```

→ `{"files": [{"name": "example", "filename": "example.wav", "size_bytes": 160044}]}`

### Example sort (origin)

```
POST /api/example-sort-origin
```

**Body:** `{"origin": {"importer": "server_folder", "params": {"path": "/data/sounds"}}, "key": "subdir/audio123.wav"}`

Sorts by similarity to a file resolved from an origin dict.

→ `{"results": [...], "threshold": 0.5123}`

### Upload server media file

```
POST /api/server-media-files/upload
```

**Form:**
- `file`: media file to upload.
- `crop_params` (optional): JSON object with the user-cropped bounds
  (audio `{"start", "end"}`, image `{"box": [...]}`). When set, the file is
  cropped server-side before being saved, so the persisted example *is* the
  cropped sub-region.
- `media_type` (required when `crop_params` is present): `"audio"` or
  `"image"`; selects which bounded clipper to apply.

→ `{"filename": "abc123.wav", "original_name": "dog_bark.wav"}` (201)

`filename` is the server-generated UUID name (the persistence key);
`original_name` is the user's file name, kept for display.
400 if the multipart body is missing a file/filename, or `crop_params` is
invalid for the given media type.

### Save loaded media as a server example file

```
POST /api/server-media-files/from-media-id
```

**Body:** `{"media_id": 42}` (optionally `{"media_id": 42, "crop_params": {...}}`
— e.g. audio `{"start", "end"}` or image `{"box": [...]}`).

Materialises a loaded media's bytes (optionally cropped) into the per-user
`example_media/` dir so the new-detector form can reference it as a seed.

→ `{"filename": "abc123.wav", "original_name": "dog_bark.wav"}` (201)

400 (media not loaded, or invalid `crop_params`), 404 (media bytes unavailable).

### Server media file thumbnail

```
GET /api/server-media-files/{filename}/thumbnail
```

Small preview image of an example file in the user's `example_media/` dir:
image bytes, an audio waveform PNG, or a video mid-frame PNG (binary, not JSON).

400 (filename escapes the media dir), 404 (not found / no thumbnail for the
type), 500 (generation failed).

### Seed importers

A **seed importer** is a plugin that contributes a *batch* of unlabeled seed
media — items that are "close but not quite" what the user is hunting for —
to a new blank detector. No seed importer ships in-tree; the family is an
extension point third-party packages register into (see
[`EXTENDING-plugins.md` § Adding a Seed Importer](../EXTENDING-plugins.md#adding-a-seed-importer)).

```
GET /api/seed-importers
```

→ `{"importers": [{"name": ..., "display_name": ..., "icon": ..., "fields": [...], "max_items": 100, ...}]}`

`{"importers": []}` on a vanilla install.

```
POST /api/seed-import/{importer_name}
```

**Body:** plugin-dependent — the fields the named importer declares (JSON, or
multipart when it declares a `file` field). Not described in the OpenAPI spec
for that reason; see [Routes absent from the spec](../API.md#routes-absent-from-the-spec).

Runs the importer and saves each returned item's bytes into the per-user
`example_media/` directory.

→
```json
{
  "items": [
    {"filename": "abc123.wav", "original_name": "near-miss-1.wav", "origin": null}
  ],
  "count": 1,
  "truncated": false
}
```

Each item plugs into the detector-example model as
`{"type": "media", "value": <filename>, "labeled": false}` — the `labeled`
flag is what keeps a seed a query rather than a Good vote (see
[Register detector](detectors.md#register-detector)). `truncated` is
`true` when the importer returned more than its `max_items` cap and the tail
was dropped. `count: 0` is a valid "nothing matched" answer, not an error.

400 (bad user input), 404 (unknown importer), 422 (missing/invalid field),
501 (`run` not implemented), 502 (upstream/source failure).

```
POST /api/seed-import/{importer_name}/options
```

Dynamic select options, same contract as the dataset-importer variant.

### Label-file sort

```
POST /api/label-file-sort
```

**Form:** `file`: JSON file with a `labels` array. Each entry has `label`
(`"good"` / `"bad"`) and a `path`/`file`/`filename` pointing to an audio file.

Trains the detector head on the labeled files, then scores all loaded medias.

→ A [windowed sort response](#sort-response-shape-windowing) plus `loaded` and
`skipped` counts for the label file:
`{"results": [...], "threshold": 0.5123, "acq_threshold": null, "sort_token": "…", "total": 10, "above_threshold": 4, "has_more_below": false, "loaded": 10, "skipped": 2}`

---

## Votes & Labels

### Get votes

```
GET /api/votes
```

→
```json
{
  "good": [0, 3, 7],
  "bad": [1, 5],
  "verified": [3],
  "click_times": {"0": 1234567890.123},
  "learned_scores": {"0": 0.9234},
  "labelset_good_count": 12,
  "labelset_bad_count": 9,
  "good_region_boxes": {"3": [0.2, 0.3, 0.55, 0.7]}
}
```

Every key is always present. `verified` lists the ids the human has explicitly
verified this session (Find mode splits verified from the unverified work
queue); it is empty outside Find mode. `labelset_good_count` /
`labelset_bad_count` are the **active detector's** persisted label counts,
which include elements that don't resolve into the loaded dataset — so they
can exceed `good.length` / `bad.length`. They fall back to the session vote
counts when no detector context is active. `good_region_boxes` maps media id
(as a string) to the normalised `[x0, y0, x1, y1]` box of a good vote cast by
drawing on the image; only good votes that carry a box appear.

### Clear votes

```
POST /api/votes/clear
```

Clears all good/bad votes without clearing the loaded dataset. Used by the
Label flow to reset votes before importing a model's labelset.

→ `{"ok": true}`

### Text-sort suggestions

```
GET /api/textsort-suggestions
```

→ `{"suggestions": ["dog barking", "cat meowing"]}`

```
POST /api/textsort-suggestions
```

**Body:** `{"text": "dog barking"}`

→ `{"ok": true}`

### Export labels

```
GET /api/labels/export
```

**Query:**
- `?goods_only=1` (optional): export only good labels.
- `?label_filter=<mode>` (optional): `good`, `bad`, `both` (default),
  `corrections` (entries where the user changed the detector's original
  label), `unverified` (Find work-queue items the human hasn't acted on), or
  `verified` (items the human has confirmed). Overrides `goods_only`. The
  session-scoped modes (`corrections` / `unverified` / `verified`) never
  include `origin_only` fallback entries.
- `?enrich=1` (optional): add per-entry `custom_metadata` and a top-level
  `available_columns` list (see the flattened `origin.params` note above).
- `?detector_name=<name>` (optional): export **that** detector's persisted
  labelset, read from its JSON file, instead of the active pair's live
  labels. Independent of `X-Dataset-Id` / `X-Detector-Id` and of any live
  Find session, so a caller naming a detector in a list (the Dashboard's row
  action) gets that detector's labels whatever the app is pointed at. The
  session-scoped `label_filter` modes are refused with 400 here — they
  partition a session this export has no part in — and an unknown name is a
  404.
- `?format=ndjson` (optional): stream the response as newline-delimited JSON
  (`application/x-ndjson`), one label entry per line, instead of the buffered
  `{"labels": [...]}` object. Use for large exports that shouldn't be
  materialised in memory server-side. The top-level `available_columns` list
  (see `enrich`) is omitted in this mode.

→ LabelSet JSON with per-element origin and MD5 info:

```json
{
  "labels": [
    {
      "origin": {"importer": "demo", "params": {"name": "esc50"}},
      "origin_name": "dog_bark_001.wav",
      "md5": "abc123...",
      "label": "good"
    }
  ]
}
```

Only `md5` and `label` are guaranteed on an entry; `origin`, `origin_name`,
`filename`, `category`, `metadata`, and `region_box` (good votes cast on a
region) appear when the underlying element has them. The export is a faithful
rendering of the **detector's** labelset, not just the session's votes: elements
that don't resolve into the active dataset are appended and marked
`"origin_only": true`. Entries where the user changed the detector's original
label carry `"is_correction": true` (never under `detector_name`, whose rows
belong to a detector the live session says nothing about).

### Import labels

```
POST /api/labels/import
```

**Body:** `{"labels": [{"origin": {...}, "origin_name": "...", "md5": "...", "label": "good"}]}`

Matches by origin+origin_name first, falls back to MD5. An entry whose `label`
is neither `"good"` nor `"bad"`, or that resolves to no loaded media, is
counted in `skipped` rather than failing the request. A `region_box` on a
`"good"` entry is round-tripped back onto the vote (ignored on `"bad"`).

→ `{"applied": 8, "skipped": 2}`

### Upload media to pile

```
POST /api/medias/add-to-pile
```

**Form:**
- `file`: the media file to upload.
- `label`: `"good"` or `"bad"`.

Uploads a media file and adds it to the Good or Bad pile. If a media with
the same MD5 already exists, the existing media is voted accordingly.
Otherwise, the file is embedded using the dataset's embedder, inserted as
a new media item, and then voted.

→ `{"ok": true, "media_id": 123, "is_new": true}` (201 if new, 200 if existing)
400 if no file, empty file, or invalid label. 400 if no dataset loaded.

---

### Fill labels from sort results

```
POST /api/labels/fill-from-sort
```

**Body:**

```json
{
  "sort_results": [{"id": 0, "score": 0.8}],
  "threshold": 0.5,
  "sides": "good",
  "confirm": false
}
```

`sides`: `"good"`, `"bad"`, or `"both"`.

When `confirm` is `false` (dry run):

→ `{"good_count": 15, "bad_count": 10}`

When `confirm` is `true`:

→ `{"good_applied": 15, "bad_applied": 10, "results": {...}}`
