import pytest

import app as app_module

# Import refactored modules and make them accessible through app_module
from config import NUM_CLIPS, SAMPLE_RATE
from vistatotes.audio import generate_wav
from vistatotes.models import initialize_models, train_and_score
from vistatotes.models.progress import clear_progress_cache
from vistatotes.utils import bad_votes, clips, good_votes, label_history

# Attach to app_module for backward compatibility with existing tests
app_module.NUM_CLIPS = NUM_CLIPS
app_module.SAMPLE_RATE = SAMPLE_RATE
app_module.generate_wav = generate_wav
app_module.train_and_score = train_and_score
app_module.clips = clips
app_module.good_votes = good_votes
app_module.bad_votes = bad_votes

# Initialize models and clips
initialize_models()
app_module.init_clips()


@pytest.fixture(autouse=True)
def reset_votes():
    """Reset vote state and progress cache before each test."""
    good_votes.clear()
    bad_votes.clear()
    label_history.clear()
    clear_progress_cache()


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
