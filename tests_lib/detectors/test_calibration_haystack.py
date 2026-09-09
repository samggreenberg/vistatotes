"""The threshold is realized on the population that will be *scored*.

The fold-anchored estimator carries its per-fold cuts to the final model in
quantile space and realizes them against one distribution.  Which distribution
that is has to be the one inference reads.  For every caller that scores the
snapshot it loaded - the GUI - the two are the same set, and ``snap`` serves as
both the label-resolution snapshot and the haystack.  The CLI converts,
re-clips and re-embeds before scoring, so it names its haystack separately
(issue #3647); these tests pin that seam.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vtscore.detectors.labelset_training as lt
import vtscore.training.thresholds as thresholds_mod
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.labelset_training import Haystack, train_from_labelset
from vtscore.state.core import DetectorContext

DIM = 8
N = 40
ID_BASE = 9000
EMBEDDER = "test_embedder"


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)


def _media(cid: int, vec: np.ndarray) -> dict:
    return {"id": cid, "media_type": "image", "embedder": EMBEDDER, "embeddings": {EMBEDDER: vec}}


@pytest.fixture
def captured_final_scores(monkeypatch):
    """The score distribution the cut is finally realized on."""
    seen: list[list[float]] = []
    real = thresholds_mod.fit_fold_anchored_cut

    def _spy(fold_haystacks, fold_orderings, final_scores, **kwargs):
        seen.append([float(s) for s in final_scores])
        return real(fold_haystacks, fold_orderings, final_scores, **kwargs)

    monkeypatch.setattr(thresholds_mod, "fit_fold_anchored_cut", _spy)
    return seen


@pytest.fixture
def stub_labels(monkeypatch):
    """Ten labels around two prototypes, cached as if their origins resolved.

    Stands in for ``populate_label_embeddings``, whose real job (fetching each
    origin's file and embedding it) is not what these tests are about; what
    matters is that it stamps ``det_ctx.embedder`` before the haystack hook is
    consulted, which the stub reproduces.
    """
    rng = np.random.default_rng(3)
    good_proto = _unit(rng.standard_normal(DIM))
    bad_proto = _unit(rng.standard_normal(DIM))

    elements = []
    vecs: dict[str, np.ndarray] = {}
    from vtscore.detectors.labelset_elements import stable_element_id

    for i in range(5):
        for label, proto in (("good", good_proto), ("bad", bad_proto)):
            elem = LabeledElement(md5=f"{label}-{i}", label=label, origin_name=f"{label}-{i}.png")
            elements.append(elem)
            vecs[stable_element_id(elem)] = _unit(proto + 0.1 * rng.standard_normal(DIM))

    def _fake_populate(det_ctx, labelset, *, media_type, snap, on_progress=None):
        det_ctx.embedder = EMBEDDER
        det_ctx.label_embeddings.update(vecs)

    monkeypatch.setattr(lt, "populate_label_embeddings", _fake_populate)
    return LabelSet(elements=elements), good_proto, rng


def _snapshots(rng, good_proto):
    """The loaded medias, and the re-clipped snapshot scoring would read.

    Every media fans out into four clips: three copies of the parent and one
    "hot" crop that looks more good-like.  Scoring maxes over a media's clips,
    so the routed distribution is by construction >= the native one - the same
    one-sided bias #3180 fixed on the region-pooling side.
    """
    native: dict[int, dict] = {}
    clips: dict[int, dict] = {}
    to_source: dict[int, int] = {}
    next_clip = 1
    for cid in range(ID_BASE, ID_BASE + N):
        img = _unit(rng.standard_normal(DIM))
        native[cid] = _media(cid, img)
        for k in range(4):
            vec = img if k < 3 else _unit(0.5 * img + 0.5 * good_proto + 0.2 * rng.standard_normal(DIM))
            clips[next_clip] = _media(next_clip, vec)
            to_source[next_clip] = cid
            next_clip += 1
    return native, clips, to_source


def _scores(model, snap) -> np.ndarray:
    from vtscore.detectors.training import score_rows_with_model, scoring_rows_for_snap

    rows = scoring_rows_for_snap(snap, EMBEDDER)
    return np.asarray(score_rows_with_model(model, rows)[0], dtype=np.float64)


class TestHaystackOverride:
    def test_cut_is_realized_on_the_haystack_not_the_snap(self, stub_labels, captured_final_scores):
        labelset, good_proto, rng = stub_labels
        native, clips, to_source = _snapshots(rng, good_proto)

        det_ctx = DetectorContext("d", media_type="image")
        assert train_from_labelset(
            det_ctx,
            labelset,
            media_type="image",
            snap=native,
            haystack_for=lambda _emb: Haystack(clips, to_source),
        )

        assert len(captured_final_scores) == 1
        fitted = np.asarray(captured_final_scores[0], dtype=np.float64)
        np.testing.assert_allclose(fitted, _scores(det_ctx.model, clips), rtol=0, atol=0)

        # ...and the two populations really are different, or this proves nothing.
        assert fitted.mean() > _scores(det_ctx.model, native).mean() + 1e-3

    def test_hook_sees_the_space_the_labels_landed_in(self, stub_labels):
        """The routing decision needs the embedder, which only exists once the
        labels are embedded - so the hook is called after, not before."""
        labelset, good_proto, rng = stub_labels
        native, clips, to_source = _snapshots(rng, good_proto)
        seen: list[str] = []

        def _hook(embedder_name: str):
            seen.append(embedder_name)
            return Haystack(clips, to_source)

        train_from_labelset(
            DetectorContext("d", media_type="image"),
            labelset,
            media_type="image",
            snap=native,
            haystack_for=_hook,
        )
        assert seen == [EMBEDDER]

    def test_no_hook_keeps_the_snap_as_the_haystack(self, stub_labels, captured_final_scores):
        """Every caller that scores what it loaded is byte-for-byte unchanged."""
        labelset, good_proto, rng = stub_labels
        native, _clips, _to_source = _snapshots(rng, good_proto)

        det_ctx = DetectorContext("d", media_type="image")
        assert train_from_labelset(det_ctx, labelset, media_type="image", snap=native)

        fitted = np.asarray(captured_final_scores[0], dtype=np.float64)
        np.testing.assert_allclose(fitted, _scores(det_ctx.model, native), rtol=0, atol=0)

    def test_hook_returning_none_keeps_the_snap(self, stub_labels, captured_final_scores):
        """An empty routed snapshot must not fit the estimator on nothing."""
        labelset, good_proto, rng = stub_labels
        native, _clips, _to_source = _snapshots(rng, good_proto)

        det_ctx = DetectorContext("d", media_type="image")
        assert train_from_labelset(
            det_ctx,
            labelset,
            media_type="image",
            snap=native,
            haystack_for=lambda _emb: None,
        )
        fitted = np.asarray(captured_final_scores[0], dtype=np.float64)
        np.testing.assert_allclose(fitted, _scores(det_ctx.model, native), rtol=0, atol=0)

    def test_threshold_moves_with_the_population(self, stub_labels):
        """The whole point: the same head cuts the scored population at the
        quantile the algorithm chose, instead of one fitted on a lower one."""
        labelset, good_proto, rng = stub_labels
        native, clips, to_source = _snapshots(rng, good_proto)

        ctx_a = DetectorContext("a", media_type="image")
        train_from_labelset(ctx_a, labelset, media_type="image", snap=native)
        ctx_b = DetectorContext("b", media_type="image")
        train_from_labelset(
            ctx_b,
            labelset,
            media_type="image",
            snap=native,
            haystack_for=lambda _emb: Haystack(clips, to_source),
        )

        # Same labels, same head - only the haystack differs.
        assert torch.allclose(ctx_a.model.state_dict()["0.weight"], ctx_b.model.state_dict()["0.weight"], atol=1e-5)

        clip_scores = _scores(ctx_b.model, clips)
        per_source: dict[int, float] = {}
        for cid, score in zip(sorted(clips), clip_scores):
            src = to_source[cid]
            per_source[src] = max(per_source.get(src, -np.inf), float(score))
        scored = np.array(sorted(per_source.values()))

        assert ctx_b.threshold > ctx_a.threshold
        # The old cut sits lower in the population it is applied to than the
        # new one - which is over-inclusion, measured as hits.
        assert int((scored >= ctx_a.threshold).sum()) > int((scored >= ctx_b.threshold).sum())


class TestVoteExclusionFollowsTheHaystack:
    def test_voted_ids_are_mapped_through_to_source(self, stub_labels, monkeypatch):
        """The #3308 exclusion names media ids; routing throws those ids away,
        so the mapping is what keeps it meaning anything."""
        labelset, good_proto, rng = stub_labels
        native, clips, to_source = _snapshots(rng, good_proto)

        # Two loaded medias carry labels.
        voted_sources = {ID_BASE, ID_BASE + 1}
        monkeypatch.setattr(lt, "labeled_media_ids", lambda _ls, _snap: set(voted_sources))

        seen: dict = {}
        import vtscore.detectors.training as training_mod

        real = training_mod.train_and_threshold

        def _spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        # ``train_from_labelset`` imports the symbol inside its own body, so
        # the module attribute is what the call resolves against.
        monkeypatch.setattr(training_mod, "train_and_threshold", _spy)

        train_from_labelset(
            DetectorContext("d", media_type="image"),
            labelset,
            media_type="image",
            snap=native,
            haystack_for=lambda _emb: Haystack(clips, to_source),
        )

        expected = {cid for cid, src in to_source.items() if src in voted_sources}
        assert seen["voted_ids"] == expected
        # Every clip of both voted media, and nothing else.
        assert len(expected) == 8
