# VectoryTones

A sound clip explorer web app. Browse any set of audio clips, listen to them in the browser, and vote clips as "good" or "bad." Supports text-based semantic sorting (via LAION-CLAP embeddings) and learned sorting (via a small neural network trained on your votes). Ships with 20 synthetic sine-wave clips by default, and includes demo datasets (audio, image, and text) that auto-download when selected from the New Dataset interface. Built with Flask (Python) and vanilla JavaScript.

## Prerequisites

You need **Python 3.9+** installed. Check by running:

```bash
python3 --version
```

If you see something like `Python 3.11.4`, you're good. If the command isn't found, install Python from [python.org/downloads](https://www.python.org/downloads/) or with your system package manager:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv

# macOS (with Homebrew)
brew install python
```

You also need **Git** to download the code:

```bash
git --version
```

If it's not installed:

```bash
# Ubuntu / Debian
sudo apt install git

# macOS (with Homebrew)
brew install git
```

## Getting the code

Open a terminal and run:

```bash
git clone https://github.com/samggreenberg/vectorytones.git
cd vectorytones
```

This downloads the project and moves you into its folder.

## Setting up a virtual environment

A virtual environment keeps this project's dependencies separate from the rest of your system. This is optional but recommended.

```bash
python3 -m venv venv
```

Then activate it:

```bash
# Linux / macOS
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
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

This installs Flask, NumPy, PyTorch, and other audio processing dependencies.

## Running the app

```bash
python app.py
```

You should see output like:

```
 * Running on http://127.0.0.1:5000
```

Open that URL in your browser. You'll see the VectoryTones interface with a list of clips on the left. Click a clip to play it and use the Good/Bad buttons to vote.

Press **Ctrl+C** in the terminal to stop the server.

## Running the tests

First, install pytest (if you haven't already):

```bash
pip install pytest
```

Then run:

```bash
python -m pytest test_app.py -v
```

You should see all 36 tests pass.

## Project structure

```
vectorytones/
├── app.py                   # Flask backend — routes, audio generation, voting, sorting
├── static/
│   └── index.html           # Frontend UI (HTML, CSS, JavaScript)
├── templates/
│   └── index.html           # Jinja2 template entry point
├── test_app.py              # Test suite (pytest)
├── requirements-cpu.txt     # CPU-only Python dependencies
├── requirements-gpu.txt     # GPU-enabled Python dependencies
├── requirements.txt         # Generic Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```
