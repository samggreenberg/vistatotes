import os

# Limit threads to reduce memory overhead in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Visual feedback for startup
print("⏳ Initializing VistaTotes...", flush=True)

import hashlib

print("⏳ Importing ML libraries (this may take a few seconds)...", flush=True)

from flask import Flask

# Import refactored modules
from config import DATA_DIR, NUM_CLIPS
from vistatotes.audio import generate_wav
from vistatotes.models import embed_audio_file, initialize_models
from vistatotes.routes import clips_bp, datasets_bp, exporters_bp, main_bp, sorting_bp
from vistatotes.utils import clips

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------


def init_clips():
    print("DEBUG: Generating synthetic waveforms...", flush=True)
    DATA_DIR.mkdir(exist_ok=True)
    temp_path = DATA_DIR / "temp_embed.wav"

    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)

        # Generate embedding by saving to temp file
        temp_path.write_bytes(wav_bytes)
        embedding = embed_audio_file(temp_path)

        clips[i] = {
            "id": i,
            "type": "audio",
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": embedding,
            "wav_bytes": wav_bytes,
        }

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()


# Model initialization is now handled by vectorytones.models.initialize_models()


# ---------------------------------------------------------------------------
# Register Blueprints
# ---------------------------------------------------------------------------

app.register_blueprint(main_bp)
app.register_blueprint(clips_bp)
app.register_blueprint(sorting_bp)
app.register_blueprint(datasets_bp)
app.register_blueprint(exporters_bp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VistaTotes \u2014 media explorer web app")
    parser.add_argument("--local", action="store_true", help="Run in local development mode")
    parser.add_argument(
        "--autodetect",
        action="store_true",
        help="Run a detector on a dataset from the command line and print predicted-Good items",
    )
    parser.add_argument("--dataset", type=str, help="Path to a dataset pickle file (used with --autodetect)")
    parser.add_argument("--detector", type=str, help="Path to a detector JSON file (used with --autodetect)")
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

    # Two-pass parsing: first pass gets --importer and --exporter names,
    # second pass adds their arguments and re-parses.
    args, remaining = parser.parse_known_args()

    importer = None
    exporter = None

    if args.autodetect and args.importer:
        from vistatotes.datasets.importers import get_importer, list_importers

        importer = get_importer(args.importer)
        if importer is None:
            available = ", ".join(imp.name for imp in list_importers())
            parser.error(f"Unknown importer: {args.importer}. Available: {available}")

        importer.add_cli_arguments(parser)

    if args.autodetect and args.exporter:
        from vistatotes.exporters import get_exporter, list_exporters

        exporter = get_exporter(args.exporter)
        if exporter is None:
            available = ", ".join(exp.name for exp in list_exporters())
            parser.error(f"Unknown exporter: {args.exporter}. Available: {available}")

        exporter.add_cli_arguments(parser)

    if importer or exporter:
        args = parser.parse_args()
    elif remaining:
        # No importer/exporter specified but there are unknown args; let
        # argparse report the error.
        parser.parse_args()

    if args.autodetect:
        # Collect exporter field values if an exporter was specified
        exporter_field_values = None
        if exporter:
            exporter_field_values = {f.key: getattr(args, f.key, f.default or None) for f in exporter.fields}

        if args.importer:
            # New importer-based path
            if not args.detector:
                parser.error("--autodetect with --importer requires --detector")

            from vistatotes.cli import autodetect_importer_main

            field_values = {f.key: getattr(args, f.key, f.default or None) for f in importer.fields}
            autodetect_importer_main(args.importer, field_values, args.detector, args.exporter, exporter_field_values)

        elif args.dataset:
            # Legacy pickle-file path
            if not args.detector:
                parser.error("--autodetect requires --detector")

            from vistatotes.cli import autodetect_main

            autodetect_main(args.dataset, args.detector, args.exporter, exporter_field_values)

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

        print("\u2705 VistaTotes is ready!", flush=True)
        print("\U0001f310 Open http://localhost:5000 in your browser", flush=True)

        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
