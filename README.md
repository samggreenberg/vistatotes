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

## Command-line interface

VTSearch provides several CLI workflows for running detectors, importing labels, and importing processors — all without starting the web server. See [CLI.md](CLI.md) for the full CLI reference.

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
├── app.py                          # Flask entry point, registers blueprints, CLI arg parsing
├── config.py                       # Constants (SAMPLE_RATE, paths, model IDs)
├── vtsearch/                       # Main application package
│   ├── cli.py                      # CLI utilities: autodetect, label import, processor import
│   ├── settings.py                 # Persistent settings & favorite processors
│   ├── routes/                     # Flask blueprints
│   │   ├── main.py                 #   Core routes
│   │   ├── clips.py                #   Clip endpoints
│   │   ├── sorting.py              #   Sorting & voting endpoints
│   │   ├── detectors.py            #   Detector endpoints
│   │   ├── datasets.py             #   Dataset management endpoints
│   │   ├── exporters.py            #   Exporter endpoints
│   │   ├── label_importers.py      #   Label importer endpoints
│   │   ├── processor_importers.py  #   Processor importer endpoints
│   │   └── settings.py             #   Settings endpoints
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
│   │   └── video/                  #   Video plugin (X-CLIP embeddings)
│   ├── datasets/                   # Dataset loading & importing
│   │   ├── loader.py               #   Dataset loading logic
│   │   ├── downloader.py           #   Demo dataset downloads
│   │   ├── config.py               #   Dataset configuration
│   │   ├── origin.py               #   Per-element provenance tracking
│   │   ├── labelset.py             #   LabelSet / LabeledElement classes
│   │   ├── ingest.py               #   Ingest missing clips from origins
│   │   └── importers/              #   Data importer plugins
│   │       ├── base.py             #     Abstract DatasetImporter base class
│   │       ├── folder/             #     Local folder importer
│   │       ├── pickle/             #     Pickle file importer
│   │       ├── http_zip/           #     HTTP archive importer (zip/tar/rar)
│   │       ├── rss_feed/           #     RSS / podcast feed importer
│   │       └── youtube_playlist/   #     YouTube playlist importer
│   ├── exporters/                  # Results exporter plugins
│   │   ├── base.py                 #   Abstract exporter base class
│   │   ├── file/                   #   JSON file exporter
│   │   ├── csv_file/               #   CSV file exporter
│   │   ├── webhook/                #   Webhook (HTTP POST) exporter
│   │   ├── email_smtp/             #   Email (SMTP) exporter
│   │   └── gui/                    #   Console / GUI exporter (default)
│   ├── labels/                     # Label management
│   │   └── importers/              #   Label importer plugins
│   │       ├── base.py             #     Abstract label importer base class
│   │       ├── json_file/          #     JSON label importer
│   │       └── csv_file/           #     CSV label importer
│   ├── processors/                 # Processor management
│   │   └── importers/              #   Processor importer plugins
│   │       ├── base.py             #     Abstract processor importer base class
│   │       ├── detector_file/      #     Load detector from JSON file
│   │       └── label_file/         #     Train detector from labeled media
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

## Machine learning

VTSearch trains a small MLP neural network on user votes to learn a binary classifier over pretrained embeddings. See [ML.md](ML.md) for full details on the model architecture, training configuration, PyTorch settings, and embedding models.

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
