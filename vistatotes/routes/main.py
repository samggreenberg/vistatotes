"""Blueprint for main application routes."""

from pathlib import Path

from flask import Blueprint, Response, current_app

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index() -> Response:
    """Serve the single-page application entry point.

    Returns:
        The ``static/index.html`` file as an HTML response.
    """
    return current_app.send_static_file("index.html")


@main_bp.route("/favicon.ico")
def favicon() -> tuple[str, int] | Response:
    """Serve the site favicon, or return a 204 No Content if it does not exist.

    Returning 204 instead of 404 suppresses repetitive browser error logs when
    no favicon has been configured.

    Returns:
        The ``static/favicon.ico`` file as a response if it exists on disk,
        otherwise an empty ``(str, int)`` tuple with HTTP status 204.
    """
    # Return empty response if file doesn't exist to stop 404 logs
    if not (Path(current_app.root_path) / "static" / "favicon.ico").exists():
        return "", 204
    return current_app.send_static_file("favicon.ico")
