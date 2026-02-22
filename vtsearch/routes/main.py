"""Blueprint for main application routes."""

from pathlib import Path

from flask import Blueprint, Response, current_app, send_from_directory

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index() -> Response:
    """Serve the single-page application entry point.

    Returns:
        The ``static/index.html`` file as an HTML response.
    """
    return current_app.send_static_file("index.html")


def _static_dir() -> Path:
    """Return the static directory path."""
    return Path(current_app.root_path) / "static"


@main_bp.route("/favicon.ico")
def favicon() -> tuple[str, int] | Response:
    """Serve the site favicon from the static directory.

    Returns:
        The ``favicon.ico`` file from ``static/`` if it exists,
        otherwise an empty ``(str, int)`` tuple with HTTP status 204.
    """
    static = _static_dir()
    if not (static / "favicon.ico").exists():
        return "", 204
    return send_from_directory(str(static), "favicon.ico", mimetype="image/x-icon")


@main_bp.route("/favicon-<variant>.ico")
def favicon_variant(variant: str) -> tuple[str, int] | Response:
    """Serve a favicon variant (smile, frown, surprised) from the static directory."""
    allowed = {"smile", "frown", "surprised"}
    if variant not in allowed:
        return "", 404
    static = _static_dir()
    filename = f"favicon-{variant}.ico"
    if not (static / filename).exists():
        return "", 204
    return send_from_directory(str(static), filename, mimetype="image/x-icon")


@main_bp.route("/logo.svg")
def logo() -> tuple[str, int] | Response:
    """Serve the site logo from the static directory.

    Returns:
        The ``logo.svg`` file from ``static/`` if it exists,
        otherwise an empty ``(str, int)`` tuple with HTTP status 204.
    """
    static = _static_dir()
    if not (static / "logo.svg").exists():
        return "", 204
    return send_from_directory(str(static), "logo.svg", mimetype="image/svg+xml")
