# Command-line interface

VTSearch provides a CLI workflow for running detectors on datasets and exporting results — all without starting the web server.

## Auto-detect (run detectors on a dataset)

Score every item in a dataset with your favorite processors (detectors) and output the items predicted as "Good."

Detectors are specified via a **settings file** (`--settings`) that lists favorite processors. Each processor is a recipe referencing a processor importer (e.g. `detector_file` for a pre-trained detector JSON, or `label_file` to train a detector from labeled media). See below for how to create one.

**From a pickle file:**

```bash
python app.py --autodetect --dataset path/to/dataset.pkl --settings settings.json
```

**From any supported data source** (folder, HTTP archive, RSS feed, YouTube playlist):

```bash
python app.py --autodetect --importer folder --path /data/sounds --media-type sounds --settings settings.json
python app.py --autodetect --importer http_archive --url https://example.com/data.zip --settings settings.json
python app.py --autodetect --importer rss_feed --url https://example.com/feed.xml --settings settings.json
python app.py --autodetect --importer youtube_playlist --url https://youtube.com/playlist?list=... --settings settings.json
```

Available importers: `folder`, `pickle`, `http_archive`, `rss_feed`, `youtube_playlist`. Each importer adds its own flags — run `python app.py --autodetect --importer <name> --help` to see them (e.g. `--max-episodes` for RSS, `--max-videos` for YouTube).

**Exporting results** — by default results are printed to the console. Add `--exporter <name>` to send them elsewhere:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter file --filepath results.json
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter csv_file --filepath results.csv
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter webhook --url https://example.com/hook
```

Available exporters: `file` (JSON), `csv_file` (CSV), `webhook` (HTTP POST), `email_smtp`, `gui` (default — print to console).

**How to get the files:**

- **Dataset file** — Export from the web UI via the dataset menu ("Export dataset"), or use a cached `.pkl` file from the `data/embeddings/` directory after loading a demo dataset.
- **Settings file** — A JSON file listing favorite processors. Each processor references a processor importer and its field values. Example:

```json
{
  "favorite_processors": [
    {
      "processor_name": "my detector",
      "processor_importer": "detector_file",
      "field_values": { "file": "path/to/detector.json" }
    }
  ]
}
```

To use a labelset (labeled media) as a detector, use the `label_file` processor importer — VTSearch will load the referenced media clips, compute their embeddings, train a model, and use that as a detector:

```json
{
  "favorite_processors": [
    {
      "processor_name": "trained from labels",
      "processor_importer": "label_file",
      "field_values": { "file": "path/to/labels.json" }
    }
  ]
}
```

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

## Development mode

Run the server bound to `0.0.0.0` for access from other devices on the network:

```bash
python app.py --local
```
