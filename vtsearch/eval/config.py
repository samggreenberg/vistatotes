"""Eval dataset configurations built from demo datasets.

Each eval dataset wraps a demo dataset and adds per-category text
descriptions — the queries a user would type in the Text Sort box.

The ``EVAL_DATASETS`` dict is keyed by demo dataset ID.  Each value is
a dict with:

- ``"demo_dataset"``: the demo dataset ID to load
- ``"queries"``: list of :class:`EvalQuery`, one per category
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalQuery:
    """One evaluation query: a text description targeting a single category."""

    text: str
    """The natural-language query to embed (what a user would type)."""

    target_category: str
    """The ground-truth category name that should rank highest."""


# ------------------------------------------------------------------
# Audio eval queries  (ESC-50)
# All S/M/L share the same 10 categories; queries are identical.
# ------------------------------------------------------------------

_SOUNDS_QUERIES = [
    EvalQuery("a dog barking", "dog"),
    EvalQuery("a cat meowing", "cat"),
    EvalQuery("a rooster crowing at dawn", "rooster"),
    EvalQuery("birds singing and chirping", "chirping_birds"),
    EvalQuery("rain falling", "rain"),
    EvalQuery("thunderstorm with loud thunder", "thunderstorm"),
    EvalQuery("a car horn honking", "car_horn"),
    EvalQuery("a chainsaw cutting wood", "chainsaw"),
    EvalQuery("crackling fire in a fireplace", "crackling_fire"),
    EvalQuery("an alarm clock ringing", "clock_alarm"),
]

# ------------------------------------------------------------------
# Image eval queries  (Caltech-101)
# All S/M/L share the same 8 categories; queries are identical.
# ------------------------------------------------------------------

_IMAGES_QUERIES = [
    EvalQuery("a photograph of a butterfly", "butterfly"),
    EvalQuery("a photograph of a dolphin", "dolphin"),
    EvalQuery("a photograph of an elephant", "elephant"),
    EvalQuery("a photograph of a grand piano", "grand_piano"),
    EvalQuery("a photograph of a helicopter", "helicopter"),
    EvalQuery("a photograph of a lobster", "lobster"),
    EvalQuery("a photograph of a starfish", "starfish"),
    EvalQuery("a photograph of a stop sign", "stop_sign"),
]

# ------------------------------------------------------------------
# Text / paragraph eval queries  (20 Newsgroups)
# All S/M/L share the same 6 categories; queries are identical.
# ------------------------------------------------------------------

_PARAGRAPHS_QUERIES = [
    EvalQuery("baseball games and athletic competition", "sports"),
    EvalQuery("outer space exploration and astronomy", "science"),
    EvalQuery("automobiles and car reviews", "cars"),
    EvalQuery("ice hockey games and NHL scores", "hockey"),
    EvalQuery("electronic circuits and components", "electronics"),
    EvalQuery("christian faith and religious practice", "religion"),
]

# ------------------------------------------------------------------
# Video eval datasets  (UCF-101)
# ------------------------------------------------------------------

_ACTIVITIES_VIDEO_QUERIES = [
    EvalQuery("someone applying eye makeup", "ApplyEyeMakeup"),
    EvalQuery("someone applying lipstick", "ApplyLipstick"),
    EvalQuery("someone brushing their teeth", "BrushingTeeth"),
    EvalQuery("a person playing drums", "Drumming"),
    EvalQuery("a person doing yo-yo tricks", "YoYo"),
]

_SPORTS_VIDEO_QUERIES = [
    EvalQuery("a person diving off a cliff", "CliffDiving"),
    EvalQuery("someone walking on their hands", "HandstandWalking"),
    EvalQuery("a person jumping rope", "JumpRope"),
    EvalQuery("someone doing push-ups", "PushUps"),
    EvalQuery("a person practicing tai chi", "TaiChi"),
]

# ------------------------------------------------------------------
# Registry — keyed by demo dataset ID
# ------------------------------------------------------------------

EVAL_DATASETS: dict[str, dict] = {
    # Audio
    "sounds_s": {
        "demo_dataset": "sounds_s",
        "queries": _SOUNDS_QUERIES,
    },
    "sounds_m": {
        "demo_dataset": "sounds_m",
        "queries": _SOUNDS_QUERIES,
    },
    "sounds_l": {
        "demo_dataset": "sounds_l",
        "queries": _SOUNDS_QUERIES,
    },
    # Image
    "images_s": {
        "demo_dataset": "images_s",
        "queries": _IMAGES_QUERIES,
    },
    "images_m": {
        "demo_dataset": "images_m",
        "queries": _IMAGES_QUERIES,
    },
    "images_l": {
        "demo_dataset": "images_l",
        "queries": _IMAGES_QUERIES,
    },
    # Text
    "paragraphs_s": {
        "demo_dataset": "paragraphs_s",
        "queries": _PARAGRAPHS_QUERIES,
    },
    "paragraphs_m": {
        "demo_dataset": "paragraphs_m",
        "queries": _PARAGRAPHS_QUERIES,
    },
    "paragraphs_l": {
        "demo_dataset": "paragraphs_l",
        "queries": _PARAGRAPHS_QUERIES,
    },
    # Video
    "activities_video": {
        "demo_dataset": "activities_video",
        "queries": _ACTIVITIES_VIDEO_QUERIES,
    },
    "sports_video": {
        "demo_dataset": "sports_video",
        "queries": _SPORTS_VIDEO_QUERIES,
    },
}
