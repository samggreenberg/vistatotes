import logging
import os
import warnings

# Limit threads to reduce memory overhead in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Suppress Werkzeug request logging (GET/POST lines) — only show errors
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Suppress HF Hub "unauthenticated requests" warning — no token needed for
# public model downloads; the warning is just noise.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*HF Hub.*")

# Visual feedback for startup
print("⏳ Initializing VTSearch...", flush=True)

import hashlib

print("⏳ Importing ML libraries (this may take a few seconds)...", flush=True)

from flask import Flask

# Import refactored modules
from config import DATA_DIR, NUM_CLIPS
from vtsearch.audio import generate_wav
from vtsearch.models import embed_audio_file, initialize_models
from vtsearch.routes import (
    clips_bp,
    datasets_bp,
    detectors_bp,
    exporters_bp,
    label_importers_bp,
    main_bp,
    processor_importers_bp,
    settings_bp,
    sorting_bp,
)
from vtsearch.media import set_progress_callback
from vtsearch.utils import clips, update_progress

# Wire media types into the Flask app's progress reporting system.
# Without this call, media types use a silent no-op callback and can run
# standalone (e.g. in a CLI tool or notebook) without Flask.
set_progress_callback(update_progress)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------


def _embedding_cache_path():
    """Path to the cached test-clip embeddings file."""
    return DATA_DIR / "clip_embedding_cache.npz"


def _embedding_cache_key():
    """Deterministic key derived from clip-generation parameters so the cache
    auto-invalidates when NUM_CLIPS, frequencies, or durations change."""
    parts = []
    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50
        dur = round(1.0 + (i % 5) * 0.5, 1)
        parts.append(f"{i}:{freq}:{dur}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def init_clips():
    import numpy as np

    DATA_DIR.mkdir(exist_ok=True)
    temp_path = DATA_DIR / "temp_embed.wav"
    cache_path = _embedding_cache_path()
    cache_key = _embedding_cache_key()

    # Try to load cached embeddings
    cached_embeddings = {}
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            if "cache_key" in data and str(data["cache_key"]) == cache_key:
                for i in range(1, NUM_CLIPS + 1):
                    k = f"emb_{i}"
                    if k in data:
                        cached_embeddings[i] = data[k]
        except Exception:
            pass

    new_embeddings = {}
    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)

        if i in cached_embeddings:
            embedding = cached_embeddings[i]
        else:
            # Generate embedding by saving to temp file
            temp_path.write_bytes(wav_bytes)
            embedding = embed_audio_file(temp_path)
            new_embeddings[i] = embedding

        fname = f"test_clip_{i}.wav"
        clips[i] = {
            "id": i,
            "type": "audio",
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": embedding,
            "wav_bytes": wav_bytes,
            "filename": fname,
            "category": "test",
            "origin": {"importer": "test", "params": {}},
            "origin_name": fname,
        }

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

    # Save cache if we computed any new embeddings (or cache didn't exist)
    if new_embeddings or not cached_embeddings:
        save_data = {"cache_key": np.array(cache_key)}
        for i in range(1, NUM_CLIPS + 1):
            save_data[f"emb_{i}"] = clips[i]["embedding"]
        np.savez(cache_path, **save_data)


# Model initialization is now handled by vtsearch.models.initialize_models()


# ---------------------------------------------------------------------------
# Register Blueprints
# ---------------------------------------------------------------------------

app.register_blueprint(main_bp)
app.register_blueprint(clips_bp)
app.register_blueprint(sorting_bp)
app.register_blueprint(detectors_bp)
app.register_blueprint(datasets_bp)
app.register_blueprint(exporters_bp)
app.register_blueprint(label_importers_bp)
app.register_blueprint(processor_importers_bp)
app.register_blueprint(settings_bp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VTSearch \u2014 media explorer web app")
    parser.add_argument("--local", action="store_true", help="Run in local development mode")
    parser.add_argument(
        "--autodetect",
        action="store_true",
        help="Run a detector on a dataset from the command line and print predicted-Good items",
    )
    parser.add_argument("--dataset", type=str, help="Path to a dataset pickle file (used with --autodetect)")
    parser.add_argument(
        "--settings",
        type=str,
        help=(
            "Path to a settings JSON file containing favorite processors. "
            "Used with --autodetect. Defaults to data/settings.json."
        ),
    )
    parser.add_argument(
        "--importer",
        type=str,
        help="Name of the data importer to use (e.g. folder, pickle, http_archive). Used with --autodetect.",
    )
    parser.add_argument(
        "--exporter",
        type=str,
        help="Name of the results exporter to use (e.g. file, email_smtp, gui). Used with --autodetect.",
    )
    parser.add_argument(
        "--import-labels",
        action="store_true",
        help="Import labels into a loaded dataset from the command line.",
    )
    parser.add_argument(
        "--label-importer",
        type=str,
        help="Name of the label importer to use (e.g. json_file, csv_file). Used with --import-labels.",
    )
    parser.add_argument(
        "--import-missing",
        choices=["yes", "no", "ask"],
        default="ask",
        help=(
            "When --import-labels finds elements not in the dataset: "
            "'yes' to auto-import from origins, 'no' to skip, 'ask' to prompt (default: ask)."
        ),
    )
    parser.add_argument(
        "--import-processor",
        action="store_true",
        help="Import a processor (detector) via a named processor importer from the command line.",
    )
    parser.add_argument(
        "--processor-importer",
        type=str,
        help="Name of the processor importer to use (e.g. detector_file, label_file). Used with --import-processor.",
    )
    parser.add_argument(
        "--processor-name",
        type=str,
        help="Name to assign to the imported processor/detector. Used with --import-processor.",
    )

    # Two-pass parsing: first pass gets --importer, --exporter,
    # --label-importer, and --processor-importer names; second pass adds
    # their arguments and re-parses.
    args, remaining = parser.parse_known_args()

    importer = None
    exporter = None
    label_importer = None
    proc_importer = None

    if args.autodetect and args.importer:
        from vtsearch.datasets.importers import get_importer, list_importers

        importer = get_importer(args.importer)
        if importer is None:
            available = ", ".join(imp.name for imp in list_importers())
            parser.error(f"Unknown importer: {args.importer}. Available: {available}")

        importer.add_cli_arguments(parser)

    if args.autodetect and args.exporter:
        from vtsearch.exporters import get_exporter, list_exporters

        exporter = get_exporter(args.exporter)
        if exporter is None:
            available = ", ".join(exp.name for exp in list_exporters())
            parser.error(f"Unknown exporter: {args.exporter}. Available: {available}")

        exporter.add_cli_arguments(parser)

    if getattr(args, "import_labels", False) and getattr(args, "label_importer", None):
        from vtsearch.labels.importers import get_label_importer, list_label_importers

        label_importer = get_label_importer(args.label_importer)
        if label_importer is None:
            available = ", ".join(imp.name for imp in list_label_importers())
            parser.error(f"Unknown label importer: {args.label_importer}. Available: {available}")

        label_importer.add_cli_arguments(parser)

    if getattr(args, "import_processor", False) and getattr(args, "processor_importer", None):
        from vtsearch.processors.importers import get_processor_importer, list_processor_importers

        proc_importer = get_processor_importer(args.processor_importer)
        if proc_importer is None:
            available = ", ".join(imp.name for imp in list_processor_importers())
            parser.error(f"Unknown processor importer: {args.processor_importer}. Available: {available}")

        proc_importer.add_cli_arguments(parser)

    if importer or exporter or label_importer or proc_importer:
        args = parser.parse_args()
    elif remaining:
        # No importer/exporter specified but there are unknown args; let
        # argparse report the error.
        parser.parse_args()

    if getattr(args, "import_processor", False):
        if not getattr(args, "processor_importer", None):
            parser.error("--import-processor requires --processor-importer <name>")

        proc_name = getattr(args, "processor_name", None) or ""
        if not proc_name.strip():
            parser.error("--import-processor requires --processor-name <name>")

        from vtsearch.cli import import_processor_main

        field_values = {f.key: getattr(args, f.key, f.default or None) for f in proc_importer.fields}
        import_processor_main(args.processor_importer, field_values, proc_name)

    elif getattr(args, "import_labels", False):
        if not getattr(args, "label_importer", None):
            parser.error("--import-labels requires --label-importer <name>")
        if not args.dataset:
            parser.error("--import-labels requires --dataset <file.pkl>")

        from vtsearch.cli import import_labels_main

        field_values = {f.key: getattr(args, f.key, f.default or None) for f in label_importer.fields}
        import_missing_flag = getattr(args, "import_missing", "ask")
        auto_import_missing = {"yes": True, "no": False, "ask": None}[import_missing_flag]
        import_labels_main(args.dataset, args.label_importer, field_values, auto_import_missing=auto_import_missing)

    elif args.autodetect:
        # Collect exporter field values if an exporter was specified
        exporter_field_values = None
        if exporter:
            exporter_field_values = {f.key: getattr(args, f.key, f.default or None) for f in exporter.fields}

        settings_path = getattr(args, "settings", None)

        if args.importer:
            # Importer-based path
            from vtsearch.cli import autodetect_importer_main

            field_values = {f.key: getattr(args, f.key, f.default or None) for f in importer.fields}
            autodetect_importer_main(
                args.importer, field_values, settings_path, args.exporter, exporter_field_values
            )

        elif args.dataset:
            # Pickle-file path
            from vtsearch.cli import autodetect_main

            autodetect_main(args.dataset, settings_path, args.exporter, exporter_field_values)

        else:
            parser.error("--autodetect requires either --dataset <file.pkl> or --importer <name>")

    elif args.local:
        # Local development mode
        print("\U0001f680 Running in LOCAL mode (accessible from other devices)", flush=True)
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        # Production mode \u2014 models load lazily when the first dataset is loaded
        print("\U0001f680 Running in PRODUCTION mode", flush=True)
        initialize_models()

        print("\u2705 VTSearch is ready!", flush=True)
        print("\U0001f310 Open http://localhost:5000 in your browser", flush=True)

        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
