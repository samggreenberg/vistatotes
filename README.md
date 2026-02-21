# VTSearch

A media explorer web app. Browse collections of audio clips, images, or text paragraphs — listen/view them in the browser and vote items as "good" or "bad." Supports text-based semantic sorting (via LAION-CLAP, CLIP, or E5-large-v2 embeddings depending on media type) and learned sorting (via a small neural network trained on your votes). Several demo datasets can be loaded directly from the UI. Built with Flask (Python) and vanilla JavaScript.

## Setup

See [SETUP.md](SETUP.md) for prerequisites, getting the code, virtual environment setup, and installing dependencies.

## Running the app

```bash
python app.py
```

You should see output like:

```
 * Running on http://127.0.0.1:5000
```

Open that URL in your browser. The app starts with no clips loaded — use the menu to load a demo dataset (see below).

Press **Ctrl+C** in the terminal to stop the server.

## Running a detector from the command line

If you have a dataset file (`.pkl`) and a detector file (`.json`), you can run the detector against the dataset without starting the web server. This prints which items the detector predicts as "Good."

```bash
python app.py --autodetect --dataset path/to/dataset.pkl --detector path/to/detector.json
```

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

Both `--dataset` and `--detector` are required when using `--autodetect`.

## Loading a demo dataset

When the app is running, click the hamburger menu in the top-left corner to open the dataset panel. From there you can browse the available demo datasets and load one. Each demo is downloaded and embedded on first use, then cached for instant loading afterward.

Available demos:

| Demo | Media type | Description |
|------|-----------|-------------|
| **sounds_s** | Audio | Animals & fireside — dogs, cats, roosters, church bells, crackling fire (ESC-50, ~200 clips) |
| **sounds_m** | Audio | Everyday sounds — babies, laughter, clapping, footsteps, chainsaws, airplanes, and more (ESC-50, ~400 clips) |
| **sounds_l** | Audio | Environmental mix — nature, weather, traffic, and household sounds (ESC-50, ~800 clips) |
| **images_s** | Image | Nature & flight — butterflies, sunflowers, starfish, helicopters (Caltech-101) |
| **images_m** | Image | Animals & objects — dolphins, pianos, elephants, kangaroos, laptops, and more (Caltech-101) |
| **images_l** | Image | Diverse objects — 15 categories including scorpions, vehicles, and instruments (Caltech-101) |
| **paragraphs_s** | Text | Sports & science articles from 20 Newsgroups |
| **paragraphs_m** | Text | World news, business, technology, and medicine articles from 20 Newsgroups |
| **paragraphs_l** | Text | Eight-topic mix — cars, hockey, electronics, crypto, religion, and more (20 Newsgroups) |
| **activities_video** | Video | Personal activities — grooming, drumming, yo-yo (UCF-101) |
| **sports_video** | Video | Sports & exercise — cliff diving, jump rope, push-ups, tai chi (UCF-101) |

You can also load your own data from pickle files or folders via the same menu.

## Running the tests

Install dev dependencies (includes pytest):

```bash
pip install -r requirements-dev.txt
```

Then run:

```bash
python -m pytest tests/ -v
```

## Project structure

```
vtsearch/
├── app.py                          # Flask entry point, registers blueprints
├── config.py                       # Constants (SAMPLE_RATE, paths, model IDs)
├── vtsearch/                     # Main application package
│   ├── routes/                     # Flask blueprints
│   │   ├── main.py                 #   Core routes
│   │   ├── clips.py                #   Clip endpoints
│   │   ├── sorting.py              #   Sorting & voting endpoints
│   │   ├── datasets.py             #   Dataset management endpoints
│   │   └── exporters.py            #   Exporter endpoints
│   ├── models/                     # ML models
│   │   ├── embeddings.py           #   Embedding model wrappers
│   │   ├── loader.py               #   Model loading
│   │   ├── training.py             #   Neural net training
│   │   └── progress.py             #   Progress tracking
│   ├── media/                      # Media type plugins
│   │   ├── base.py                 #   Abstract MediaType base class
│   │   ├── audio/                  #   Audio plugin (LAION-CLAP embeddings)
│   │   ├── image/                  #   Image plugin (CLIP embeddings)
│   │   ├── text/                   #   Text plugin (E5-large-v2 embeddings)
│   │   └── video/                  #   Video plugin
│   ├── datasets/                   # Dataset loading & importing
│   │   ├── loader.py               #   Dataset loading logic
│   │   ├── downloader.py           #   Demo dataset downloads
│   │   ├── config.py               #   Dataset configuration
│   │   └── importers/              #   Data importer plugins
│   │       ├── base.py             #     Abstract DatasetImporter base class
│   │       ├── folder/             #     Folder importer
│   │       ├── pickle/             #     Pickle file importer
│   │       └── http_zip/           #     HTTP ZIP archive importer
│   ├── exporters/                  # Results exporter plugins
│   │   ├── base.py                 #   Abstract exporter base class
│   │   ├── email_smtp/             #   Email (SMTP) exporter
│   │   ├── file/                   #   File exporter
│   │   └── gui/                    #   GUI exporter
│   ├── audio/                      # Audio utilities
│   │   └── generator.py            #   Audio generation
│   └── utils/                      # Shared utilities
│       ├── state.py                #   Global state (clips, votes)
│       └── progress.py             #   Progress helpers
├── static/                         # Frontend
│   ├── index.html                  #   HTML structure
│   ├── app.js                      #   All frontend JavaScript
│   └── styles.css                  #   All CSS styles
├── tests/                          # Test suite (pytest)
├── requirements.txt                # Core Python dependencies
├── requirements-cpu.txt            # CPU-only dependencies (PyTorch CPU wheel)
├── requirements-gpu.txt            # GPU-enabled dependencies (PyTorch with CUDA)
├── requirements-dev.txt            # Dev dependencies (requirements.txt + pytest)
├── requirements-importers.txt      # Aggregated importer dependencies
├── requirements-exporters.txt      # Aggregated exporter dependencies
├── EVAL.md                         # Evaluation guide
├── EXTENDING.md                    # Guide for writing plugins
└── README.md
```

## Evaluation

VTSearch includes an evaluation framework that measures sorting quality on demo datasets. Run it with:

```bash
python -m vtsearch.eval --plot-dir eval_output
```

This runs text-sort and learned-sort evaluations across all demo datasets, prints a summary, and saves visualisation charts as PNGs. See [EVAL.md](EVAL.md) for the full guide, including:

- **[CLI reference](EVAL.md#cli-reference)** — All flags and options for the eval runner.
- **[Understanding the metrics](EVAL.md#understanding-the-metrics)** — What mAP, P@k, R@k, F1, and other metrics mean.
- **[Visualisations](EVAL.md#visualisations)** — Charts generated by the eval framework.
- **[Writing a custom evaluation script](EVAL.md#writing-a-custom-evaluation-script)** — How to sweep over parameters, run voting-iteration simulations, and use the Python API directly.

## Extending with plugins

VTSearch has a plugin architecture for media types, data importers, and results exporters. See [EXTENDING.md](EXTENDING.md) for full documentation, including:

- **[Adding a Data Importer](EXTENDING.md#adding-a-data-importer)** — Auto-discovered plugins that load datasets from new sources (S3, databases, APIs, etc.). Subclass `DatasetImporter`, expose an `IMPORTER` instance, and the system wires up API routes and UI forms automatically.
- **[Adding a Results Exporter](EXTENDING.md#adding-a-results-exporter)** — Export votes, labels, or detector weights in new formats by adding routes to the appropriate blueprint.
- **[Adding a Media Type](EXTENDING.md#adding-a-media-type)** — Support new content types (code, 3D models, etc.) by subclassing `MediaType` with embedding, serving, and clip-loading methods.
- **[Dependency Management](EXTENDING.md#dependency-management)** — How the layered requirements file structure works and where to add new dependencies.
