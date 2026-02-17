"""Blueprint for main application routes."""

from pathlib import Path

from flask import Blueprint, current_app

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return current_app.send_static_file("index.html")


@main_bp.route("/favicon.ico")
def favicon():
    # Return empty response if file doesn't exist to stop 404 logs
    if not (Path(current_app.root_path) / "static" / "favicon.ico").exists():
        return "", 204
    return current_app.send_static_file("favicon.ico")
