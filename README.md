# VTSearch

A media explorer web app. Browse collections of audio clips, images, or text paragraphs — listen/view them in the browser and vote items as "good" or "bad." Supports text-based semantic sorting (via LAION-CLAP, CLIP, or E5-large-v2 embeddings depending on media type) and learned sorting (via a small neural network trained on your votes). Several demo datasets can be loaded directly from the UI. Built with Flask (Python) and vanilla JavaScript.

## Setup

See [docs/SETUP.md](docs/SETUP.md) for prerequisites, getting the code, virtual environment setup, and installing dependencies.

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

VTSearch provides several CLI workflows for running detectors, importing labels, and importing processors — all without starting the web server. See [docs/CLI.md](docs/CLI.md) for the full CLI reference.

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
├── app.py                          # Flask entry point, registers blueprints, CLI arg parsing
├── vtsearch/                       # Main application package
│   ├── config.py                   # Constants (SAMPLE_RATE, paths, model IDs)
│   ├── clips.py                    # Test clip generation & embedding cache
│   ├── cli.py                      # CLI utilities: autodetect, label import, processor import
│   ├── settings.py                 # Persistent settings & favorite processors
│   ├── routes/                     # Flask blueprints
│   ├── models/                     # ML models (embeddings, training, progress)
│   ├── media/                      # Media type plugins (audio, image, text, video)
│   ├── datasets/                   # Dataset loading, downloading, importers
│   ├── exporters/                  # Results exporter plugins
│   ├── labels/importers/           # Label importer plugins
│   ├── processors/importers/       # Processor importer plugins
│   ├── audio/                      # Audio generation utility
│   └── utils/                      # Global state (clips, votes) & progress helpers
├── static/                         # Frontend (HTML, JS, CSS, assets)
├── tests/                          # Test suite (pytest)
├── docs/                           # Extended documentation
│   ├── ARCHITECTURE.md             # Architecture deep-dive
│   ├── EXTENDING.md                # Plugin authoring guide
│   ├── EVAL.md                     # Evaluation framework guide
│   ├── CLI.md                      # CLI reference
│   ├── ML.md                       # ML model details
│   └── SETUP.md                    # Setup instructions
└── requirements*.txt               # Dependency files (cpu, gpu, dev, importers, exporters)
```

## Machine learning

VTSearch trains a small MLP neural network on user votes to learn a binary classifier over pretrained embeddings. See [docs/ML.md](docs/ML.md) for full details on the model architecture, training configuration, PyTorch settings, and embedding models.

## Evaluation

VTSearch includes an evaluation framework that measures sorting quality on demo datasets. Run it with:

```bash
python -m vtsearch.eval --plot-dir eval_output
```

This runs text-sort and learned-sort evaluations across all demo datasets, prints a summary, and saves visualisation charts as PNGs. See [docs/EVAL.md](docs/EVAL.md) for the full guide, including:

- **[CLI reference](docs/EVAL.md#cli-reference)** — All flags and options for the eval runner.
- **[Understanding the metrics](docs/EVAL.md#understanding-the-metrics)** — What mAP, P@k, R@k, F1, and other metrics mean.
- **[Visualisations](docs/EVAL.md#visualisations)** — Charts generated by the eval framework.
- **[Writing a custom evaluation script](docs/EVAL.md#writing-a-custom-evaluation-script)** — How to sweep over parameters, run voting-iteration simulations, and use the Python API directly.

## Extending with plugins

VTSearch has a plugin architecture for media types, data importers, and results exporters. See [docs/EXTENDING.md](docs/EXTENDING.md) for full documentation, including:

- **[Adding a Data Importer](docs/EXTENDING.md#adding-a-data-importer)** — Auto-discovered plugins that load datasets from new sources (S3, databases, APIs, etc.). Subclass `DatasetImporter`, expose an `IMPORTER` instance, and the system wires up API routes and UI forms automatically.
- **[Adding a Results Exporter](docs/EXTENDING.md#adding-a-results-exporter)** — Export votes, labels, or detector weights in new formats by adding routes to the appropriate blueprint.
- **[Adding a Media Type](docs/EXTENDING.md#adding-a-media-type)** — Support new content types (code, 3D models, etc.) by subclassing `MediaType` with embedding, serving, and clip-loading methods.
- **[Dependency Management](docs/EXTENDING.md#dependency-management)** — How the layered requirements file structure works and where to add new dependencies.
