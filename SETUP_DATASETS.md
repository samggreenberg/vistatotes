# VectoryTones Dataset Setup

This guide explains how to set up audio datasets with LAION-CLAP embeddings for the VectoryTones application.

## Overview

The setup script downloads the ESC-50 environmental sound dataset, generates semantic audio embeddings using LAION-CLAP, and organizes clips into themed datasets.

## Prerequisites

Install the required dependencies:

```bash
cd ~/vectorytones
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Setup

Execute the setup script:

```bash
python setup_datasets.py
```

This will:
1. Download ESC-50 dataset (~600 MB) from GitHub
2. Extract 2,000 audio clips (50 categories, 40 clips each)
3. Load the pretrained LAION-CLAP model
4. Generate 512-dimensional embeddings for each audio clip
5. Organize clips into 4 themed datasets:
   - **animals**: Animal sounds (dogs, cats, birds, frogs, etc.)
   - **natural**: Nature sounds (rain, wind, water, fire, etc.)
   - **urban**: City sounds (cars, sirens, construction, etc.)
   - **household**: Home sounds (appliances, doors, footsteps, etc.)

## Output Structure

```
data/
├── esc50.zip                    # Downloaded archive
├── ESC-50-master/               # Extracted audio files
│   └── audio/                   # 2,000 .wav files
└── embeddings/                  # Processed datasets
    ├── animals.pkl              # Pickled dataset with embeddings
    ├── animals_info.json        # Human-readable metadata
    ├── natural.pkl
    ├── natural_info.json
    ├── urban.pkl
    ├── urban_info.json
    ├── household.pkl
    └── household_info.json
```

## Dataset Format

Each `.pkl` file contains:
```python
{
    'name': 'dataset_name',
    'clips': {
        clip_id: {
            'id': int,
            'filename': str,
            'category': str,  # ESC-50 category
            'duration': float,  # seconds
            'file_size': int,  # bytes
            'embedding': list[float]  # 512-dim LAION-CLAP vector
        }
    },
    'audio_dir': str  # Path to audio files
}
```

## Using the Datasets

After setup, start the Flask application:

```bash
python app.py
```

The app will automatically load all available datasets and default to the "animals" dataset.

### API Endpoints

- `GET /api/datasets` - List all available datasets
- `POST /api/datasets/<name>/select` - Switch to a different dataset
- `GET /api/clips` - List clips in current dataset
- `GET /api/clips/<id>` - Get clip details including embedding
- `GET /api/clips/<id>/audio` - Stream audio file
- `POST /api/clips/<id>/vote` - Vote on a clip (good/bad)
- `GET /api/votes` - Get current votes
- `GET /api/status` - Application status

### Example Usage

```bash
# List available datasets
curl http://localhost:5000/api/datasets

# Switch to natural sounds
curl -X POST http://localhost:5000/api/datasets/natural/select

# Get clips
curl http://localhost:5000/api/clips

# Get embedding for clip #1
curl http://localhost:5000/api/clips/1
```

## Dataset Statistics

After setup, you'll have approximately:
- **animals**: 40-60 clips
- **natural**: 40-60 clips
- **urban**: 60-80 clips
- **household**: 60-80 clips

Each clip is a 5-second audio file with a 512-dimensional LAION-CLAP embedding.

## Notes

- The setup script only needs to be run once
- Audio files are NOT included in the git repository
- Embeddings are computed once and saved for fast loading
- The LAION-CLAP model is downloaded automatically on first run
- ESC-50 audio is sampled at 44.1kHz, resampled to 48kHz for CLAP

## License

ESC-50 dataset: Creative Commons Attribution Non-Commercial (CC BY-NC)
LAION-CLAP model: MIT License

## References

- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50)
- [LAION-CLAP](https://github.com/LAION-AI/CLAP)
- [PyPI: laion-clap](https://pypi.org/project/laion-clap/)
