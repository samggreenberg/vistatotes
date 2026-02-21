# Setup Guide

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
git clone https://github.com/samggreenberg/vtsearch.git
cd vtsearch
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
