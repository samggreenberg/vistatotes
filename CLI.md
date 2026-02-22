# Command-line interface

VTSearch provides several CLI workflows for running detectors, importing labels, and importing processors — all without starting the web server.

## Auto-detect (run a detector on a dataset)

Score every item in a dataset with a trained detector and output the items predicted as "Good."

**From a pickle file:**

```bash
python app.py --autodetect --dataset path/to/dataset.pkl --detector path/to/detector.json
```

**From any supported data source** (folder, HTTP archive, RSS feed, YouTube playlist):

```bash
python app.py --autodetect --importer folder --path /data/sounds --media-type sounds --detector detector.json
python app.py --autodetect --importer http_archive --url https://example.com/data.zip --detector detector.json
python app.py --autodetect --importer rss_feed --url https://example.com/feed.xml --detector detector.json
python app.py --autodetect --importer youtube_playlist --url https://youtube.com/playlist?list=... --detector detector.json
```

Available importers: `folder`, `pickle`, `http_archive`, `rss_feed`, `youtube_playlist`. Each importer adds its own flags — run `python app.py --autodetect --importer <name> --help` to see them (e.g. `--max-episodes` for RSS, `--max-videos` for YouTube).

**Exporting results** — by default results are printed to the console. Add `--exporter <name>` to send them elsewhere:

```bash
python app.py --autodetect --dataset data.pkl --detector detector.json --exporter file --filepath results.json
python app.py --autodetect --dataset data.pkl --detector detector.json --exporter csv --filepath results.csv
python app.py --autodetect --dataset data.pkl --detector detector.json --exporter webhook --url https://example.com/hook
```

Available exporters: `file` (JSON), `csv` (CSV), `webhook` (HTTP POST), `email_smtp`, `gui` (default — print to console).

**How to get the files:**

- **Dataset file** — Export from the web UI via the dataset menu ("Export dataset"), or use a cached `.pkl` file from the `data/embeddings/` directory after loading a demo dataset.
- **Detector file** — In the web UI, vote on some items, then export a detector from the sorting panel. Save the returned JSON to a file. You can also use a favorite detector exported via the API (`POST /api/detector/export`).

**Example output:**

```
Predicted Good (5 items):

  1-34094-A-6.wav  (score: 0.9832, category: cat)
  1-30226-A-0.wav  (score: 0.9541, category: dog)
  1-17150-B-2.wav  (score: 0.8923, category: cat)
  1-22694-A-4.wav  (score: 0.7612, category: dog)
  1-77445-A-1.wav  (score: 0.6204, category: cat)
```

## Import labels

Apply voting labels (good/bad) to items in an existing dataset. Useful for batch-labeling from an external source.

```bash
python app.py --import-labels --dataset data.pkl --label-importer json_file --file labels.json
python app.py --import-labels --dataset data.pkl --label-importer csv_file --file labels.csv
```

Available label importers: `json_file`, `csv_file`.

When the label file references items not present in the dataset, you are prompted whether to import them from their origins. Use `--import-missing yes|no|ask` to control this (default: `ask`).

## Import a processor (detector)

Import or train a detector from the command line and save it as a favorite processor for later use.

**Load a pre-trained detector from JSON:**

```bash
python app.py --import-processor --processor-importer detector_file --processor-name "my detector" --file detector.json
```

**Train a new detector from labeled media files:**

```bash
python app.py --import-processor --processor-importer label_file --processor-name "trained" --file labels.json --media-type audio
```

Available processor importers: `detector_file`, `label_file`.

## Development mode

Run the server bound to `0.0.0.0` for access from other devices on the network:

```bash
python app.py --local
```
