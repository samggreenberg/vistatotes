# VistaTotes

A media explorer web app. Browse collections of audio clips, images, or text paragraphs — listen/view them in the browser and vote items as "good" or "bad." Supports text-based semantic sorting (via LAION-CLAP, CLIP, or E5-large-v2 embeddings depending on media type) and learned sorting (via a small neural network trained on your votes). Several demo datasets can be loaded directly from the UI. Built with Flask (Python) and vanilla JavaScript.

## Prerequisites

You need **Python 3.9+** installed. Check by running:

```bash
python3 --version
```

If you see something like `Python 3.11.4`, you're good. If the command isn't found, install Python from [python.org/downloads](https://www.python.org/downloads/) or with your system package manager.

Ubuntu / Debian:

```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv
```

macOS (with Homebrew):

```bash
brew install python
```

You also need **Git** to download the code:

```bash
git --version
```

If it's not installed:

Ubuntu / Debian:

```bash
sudo apt install git
```

macOS (with Homebrew):

```bash
brew install git
```

## Getting the code

```bash
git clone https://github.com/samggreenberg/vistatotes.git
cd vistatotes
```

## Setting up a virtual environment

A virtual environment keeps this project's dependencies separate from the rest of your system. This is optional but recommended.

```bash
python3 -m venv venv
```

Then activate it:

Linux / macOS:

```bash
source venv/bin/activate
```

Windows (Command Prompt):

```bat
venv\Scripts\activate.bat
```

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
```

When activated, you'll see `(venv)` at the start of your terminal prompt.

## Installing dependencies

Choose the appropriate requirements file based on your system:

**For CPU only** (recommended if you don't have a compatible GPU):

```bash
pip install -r requirements-cpu.txt
```

**For GPU** (NVIDIA CUDA-compatible systems):

```bash
pip install -r requirements-gpu.txt
```

This installs Flask, NumPy, PyTorch, and other ML / media processing dependencies.

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

## Loading a demo dataset

When the app is running, click the hamburger menu in the top-left corner to open the dataset panel. From there you can browse the available demo datasets and load one. Each demo is downloaded and embedded on first use, then cached for instant loading afterward.

Available demos:

| Demo | Media type | Description |
|------|-----------|-------------|
| **animals** | Audio | Animal and nature sounds (ESC-50) |
| **natural** | Audio | Natural environmental sounds (ESC-50) |
| **urban** | Audio | Urban and mechanical sounds (ESC-50) |
| **household** | Audio | Household and human sounds (ESC-50) |
| **objects_images** | Image | Common objects and animals (CIFAR-10) |
| **news_paragraphs** | Text | News article paragraphs (20 Newsgroups) |

You can also load your own data from pickle files or folders via the same menu.

## Running the tests

Install dev dependencies (includes pytest):

```bash
pip install -r requirements-dev.txt
```

Then run:

```bash
python -m pytest test_app.py -v
```

## Project structure

```
vistatotes/
├── app.py                   # Flask backend — routes, embedding models, voting,
│                            #   sorting (text, learned, detector, example, label-file),
│                            #   dataset management, and demo dataset downloads
├── static/
│   └── index.html           # Single-page frontend (HTML, CSS, vanilla JS)
├── test_app.py              # Test suite (pytest)
├── requirements.txt         # Core Python dependencies
├── requirements-cpu.txt     # CPU-only dependencies (PyTorch CPU wheel)
├── requirements-gpu.txt     # GPU-enabled dependencies (PyTorch with CUDA)
├── requirements-dev.txt     # Dev dependencies (requirements.txt + pytest)
├── TEST_COVERAGE_ANALYSIS.md
├── .gitignore
└── README.md
```
