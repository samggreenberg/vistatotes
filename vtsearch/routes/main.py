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


def _project_root() -> Path:
    """Return the project root directory (parent of the ``vtsearch`` package)."""
    return Path(current_app.root_path).parent


@main_bp.route("/favicon.ico")
def favicon() -> tuple[str, int] | Response:
    """Serve the site favicon from the project root.

    Returns:
        The ``favicon.ico`` file from the project root if it exists,
        otherwise an empty ``(str, int)`` tuple with HTTP status 204.
    """
    root = _project_root()
    if not (root / "favicon.ico").exists():
        return "", 204
    return send_from_directory(str(root), "favicon.ico", mimetype="image/x-icon")


@main_bp.route("/logo.svg")
def logo() -> tuple[str, int] | Response:
    """Serve the site logo from the project root.

    Returns:
        The ``logo.svg`` file from the project root if it exists,
        otherwise an empty ``(str, int)`` tuple with HTTP status 204.
    """
    root = _project_root()
    if not (root / "logo.svg").exists():
        return "", 204
    return send_from_directory(str(root), "logo.svg", mimetype="image/svg+xml")
