# Test Coverage Analysis

## Current State

| Metric | Value |
|--------|-------|
| Total tests | 77 |
| Passing | 31 |
| Failing | 46 |
| Statement coverage | 23% (203/896 statements in `app.py`) |

The 46 test failures stem from two breaking changes in `app.py` that the tests were never updated for:

1. **`good_votes`/`bad_votes` changed from `set` to `dict`** — Tests still use set APIs like `.update({1, 2})` and `.add(1)`, which fail on the `dict[int, None]` type now used.
2. **`initialize_app()` no longer auto-loads synthetic clips** — The app now waits for the user to load a dataset via the UI, but the test module calls `initialize_app()` at import time and expects 20 synthetic clips to be pre-populated.

These are straightforward to fix (call `init_clips()` in test setup, update vote manipulation to use dict syntax) but represent the first priority.

---

## Coverage Gaps By Category

### 1. Broken Test Infrastructure (Priority: Critical)

Fix the 46 failing tests to restore the working baseline. Specific issues:

- **`test_app.py:10`** — `initialize_app()` no longer populates `clips`. Tests need to also call `init_clips()` or the fixture must populate synthetic clips.
- **`test_app.py:14-17`** — `reset_votes` fixture uses `.clear()` on dicts, which works, but the test bodies call `good_votes.update({1, 2})` passing a set where a dict of `{1: None, 2: None}` is expected. Every test that manipulates votes needs updating.

### 2. Untested API Routes (Priority: High)

10 of 24 Flask routes have zero test coverage:

| Route | Method | Function | Lines |
|-------|--------|----------|-------|
| `/api/clips/<id>/video` | GET | `clip_video` | 927-949 |
| `/api/inclusion` | GET | `get_inclusion` | 1328-1330 |
| `/api/inclusion` | POST | `set_inclusion` | 1334-1350 |
| `/api/detector/export` | POST | `export_detector` | 1354-1388 |
| `/api/detector-sort` | POST | `detector_sort` | 1392-1435 |
| `/api/example-sort` | POST | `example_sort` | 1438-1488 |
| `/api/label-file-sort` | POST | `label_file_sort` | 1491-1612 |
| `/api/dataset/status` | GET | `dataset_status` | 1620-1629 |
| `/api/dataset/progress` | GET | `dataset_progress` | 1632-1636 |
| `/api/dataset/demo-list` | GET | `demo_dataset_list` | 1639-1693 |
| `/api/dataset/load-demo` | POST | `load_demo_dataset_route` | 1696-1718 |
| `/api/dataset/load-file` | POST | `load_dataset_file` | 1721-1751 |
| `/api/dataset/load-folder` | POST | `load_dataset_folder` | 1754-1780 |
| `/api/dataset/export` | GET | `export_dataset` | 1783-1798 |
| `/api/dataset/clear` | POST | `clear_dataset_route` | 1801-1805 |

These are the most impactful additions because they test user-facing behavior with no model dependency beyond what `init_clips()` provides.

**Recommended tests to add:**

- **Inclusion endpoints**: GET returns default 0, POST updates value, POST clamps to [-10, +10], POST rejects non-numeric input.
- **Video endpoint**: Returns 404 for missing clip, returns 400 for audio-only clip, returns correct mimetype by extension.
- **Dataset status/progress/demo-list**: These are pure reads of global state — easy to test, high value.
- **Dataset clear**: Verify clips and votes are emptied.
- **Detector export**: Train on votes, verify returned JSON has `weights` and `threshold` keys.
- **Detector sort**: Export a detector, then use it to sort — full roundtrip test.

### 3. ML Functions (Priority: High)

These core functions have no direct unit tests:

| Function | Lines | Why It Matters |
|----------|-------|----------------|
| `calculate_gmm_threshold()` | 980-1011 | Used by text sort and example sort; determines the accept/reject line |
| `find_optimal_threshold()` | 1111-1169 | Used by cross-calibration; FPR/FNR cost logic |
| `calculate_cross_calibration_threshold()` | 1172-1215 | Used by learned sort and detector export |
| `train_model()` | 1052-1108 | Core model training; inclusion weighting logic |

**Recommended tests:**

- **`calculate_gmm_threshold`**: Test with bimodal data (two clear clusters), unimodal data, single-element input, empty input. Verify threshold falls between cluster means.
- **`find_optimal_threshold`**: Test with perfectly separable scores/labels, overlapping scores, different inclusion values (positive biases toward inclusion, negative toward exclusion).
- **`calculate_cross_calibration_threshold`**: Test with small datasets (n < 4 returns 0.5), larger datasets return a reasonable float.
- **`train_model`**: Verify output model is in eval mode, output is a `nn.Module`, predictions are in [0, 1], inclusion weighting shifts predictions.

### 4. Dataset Management Functions (Priority: Medium)

These functions are completely untested:

| Function | Lines | Notes |
|----------|-------|-------|
| `update_progress()` | 302-309 | Thread-safe progress tracker |
| `clear_dataset()` | 312-317 | Clears all state |
| `load_dataset_from_pickle()` | 320-383 | Handles old and new pickle formats |
| `embed_audio_file()` | 386-411 | CLAP embedding for single file |
| `embed_video_file()` | 414-459 | VideoMAE embedding for single file |
| `load_dataset_from_folder()` | 462-555 | Loads arbitrary folder of media |
| `download_file_with_progress()` | 558-571 | HTTP download with progress |
| `download_esc50()` | 574-597 | ESC-50 dataset download |
| `load_esc50_metadata()` | 600-615 | CSV metadata parser |
| `download_ucf101_subset()` | 618-641 | UCF-101 video download |
| `load_video_metadata_from_folders()` | 644-664 | Video folder scanning |
| `load_demo_dataset()` | 667-853 | Full demo dataset pipeline |
| `export_dataset_to_file()` | 856-881 | Dataset export to bytes |

**Recommended tests (mockable without real downloads):**

- **`update_progress`**: Set values, verify `progress_data` reflects them, test thread safety.
- **`clear_dataset`**: Populate state, clear, verify empty.
- **`load_dataset_from_pickle`**: Create a minimal pickle with the expected format, load it, verify clips are populated correctly. Test both old format (bare dict) and new format (dict with `"clips"` key).
- **`export_dataset_to_file`**: Populate clips, export, verify the resulting bytes are valid pickle, re-load and verify roundtrip.
- **`load_esc50_metadata`**: Create a temp CSV in ESC-50 format, verify parsed dict.
- **`load_video_metadata_from_folders`**: Create temp directory structure, verify metadata extraction.

### 5. Edge Cases and Error Paths (Priority: Medium)

The existing tests mostly cover the happy path. Notable missing edge cases:

- **Vote on clip after dataset clear** — What happens when clips dict is empty but votes reference old IDs?
- **Sort with empty clip set** — `/api/sort` and `/api/learned-sort` when no clips are loaded.
- **Detector sort with mismatched embedding dimensions** — weights from one dataset applied to another.
- **`label_file_sort` with malformed JSON** — The function uses `eval()` on user input (`app.py:1509`), which is a security concern that should at minimum be tested.
- **Import labels with empty list** — `POST /api/labels/import` with `{"labels": []}`.
- **Concurrent progress updates** — `progress_lock` is used but never tested under contention.

### 6. No Frontend Tests (Priority: Low)

There are no JavaScript tests for the frontend (`static/index.html`). This file contains significant client-side logic. Adding even basic smoke tests with a headless browser would catch regressions in the UI layer, but this is a larger investment.

---

## Recommended Action Plan

### Phase 1: Fix Existing Tests

Update `test_app.py` to work with the current `app.py`:
- Add `init_clips()` call in test setup so synthetic clips are available.
- Change all `good_votes.update({...})` and `good_votes.add(...)` calls to use dict-compatible syntax (`good_votes.update({id: None for id in [1,2]})` or `good_votes[id] = None`).
- This alone would restore 77 passing tests and raise coverage from 23% to ~40%.

### Phase 2: Add Tests for Untested Routes

Add test classes for the 15 untested API endpoints, focusing on:
- Inclusion GET/POST (simple, no ML dependency)
- Dataset status/progress/demo-list/clear (read-only or simple state mutation)
- Video clip endpoint (needs a clip with `type: "video"`)
- Detector export and detector-sort (roundtrip test)

Estimated impact: coverage to ~55-60%.

### Phase 3: Add ML Unit Tests

Add direct tests for `calculate_gmm_threshold`, `find_optimal_threshold`, `calculate_cross_calibration_threshold`, and `train_model` with synthetic data. These don't need real audio — just numpy arrays of the right shape.

Estimated impact: coverage to ~65-70%.

### Phase 4: Add Dataset Function Tests

Use mocks and temp files to test `load_dataset_from_pickle`, `export_dataset_to_file`, `load_esc50_metadata`, and `load_video_metadata_from_folders`.

Estimated impact: coverage to ~75%.

---

## Security Note

`app.py:1509` uses `eval()` to parse user-uploaded label files:
```python
label_data = eval(text) if text.strip().startswith("{") else None
```
This is a code execution vulnerability. It should be replaced with `json.loads()` exclusively. A test should verify that malicious input (e.g., `__import__('os').system('...')`) does not execute.
