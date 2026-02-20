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
# Audio eval datasets  (ESC-50)
# ------------------------------------------------------------------

_NATURE_SOUNDS_QUERIES = [
    EvalQuery("birds singing and chirping", "chirping_birds"),
    EvalQuery("a crow cawing", "crow"),
    EvalQuery("frogs croaking near a pond", "frog"),
    EvalQuery("buzzing insects", "insects"),
    EvalQuery("rain falling", "rain"),
    EvalQuery("ocean waves crashing on shore", "sea_waves"),
    EvalQuery("thunderstorm with loud thunder", "thunderstorm"),
    EvalQuery("strong wind blowing", "wind"),
    EvalQuery("dripping water drops", "water_drops"),
    EvalQuery("crickets chirping at night", "crickets"),
]

_CITY_SOUNDS_QUERIES = [
    EvalQuery("a car horn honking", "car_horn"),
    EvalQuery("emergency siren wailing", "siren"),
    EvalQuery("engine running and revving", "engine"),
    EvalQuery("a train passing by on rails", "train"),
    EvalQuery("helicopter flying overhead", "helicopter"),
    EvalQuery("vacuum cleaner running", "vacuum_cleaner"),
    EvalQuery("washing machine spinning", "washing_machine"),
    EvalQuery("an alarm clock ringing", "clock_alarm"),
    EvalQuery("someone typing on a keyboard", "keyboard_typing"),
    EvalQuery("knocking on a wooden door", "door_wood_knock"),
]

# ------------------------------------------------------------------
# Image eval datasets  (CIFAR-10)
# ------------------------------------------------------------------

_ANIMALS_IMAGES_QUERIES = [
    EvalQuery("a photograph of a bird", "bird"),
    EvalQuery("a photograph of a cat", "cat"),
    EvalQuery("a photograph of a deer", "deer"),
    EvalQuery("a photograph of a dog", "dog"),
    EvalQuery("a photograph of a frog", "frog"),
    EvalQuery("a photograph of a horse", "horse"),
]

_VEHICLES_IMAGES_QUERIES = [
    EvalQuery("a photograph of an airplane", "airplane"),
    EvalQuery("a photograph of a car", "automobile"),
    EvalQuery("a photograph of a ship", "ship"),
    EvalQuery("a photograph of a truck", "truck"),
]

# ------------------------------------------------------------------
# Text / paragraph eval datasets  (20 Newsgroups)
# ------------------------------------------------------------------

_WORLD_NEWS_QUERIES = [
    EvalQuery("international politics and world affairs", "world"),
    EvalQuery("buying and selling goods and products", "business"),
]

_SPORTS_SCIENCE_QUERIES = [
    EvalQuery("baseball games and athletic competition", "sports"),
    EvalQuery("outer space exploration and astronomy", "science"),
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
    "nature_sounds": {
        "demo_dataset": "nature_sounds",
        "queries": _NATURE_SOUNDS_QUERIES,
    },
    "city_sounds": {
        "demo_dataset": "city_sounds",
        "queries": _CITY_SOUNDS_QUERIES,
    },
    # Image
    "animals_images": {
        "demo_dataset": "animals_images",
        "queries": _ANIMALS_IMAGES_QUERIES,
    },
    "vehicles_images": {
        "demo_dataset": "vehicles_images",
        "queries": _VEHICLES_IMAGES_QUERIES,
    },
    # Text
    "world_news": {
        "demo_dataset": "world_news",
        "queries": _WORLD_NEWS_QUERIES,
    },
    "sports_science_news": {
        "demo_dataset": "sports_science_news",
        "queries": _SPORTS_SCIENCE_QUERIES,
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
