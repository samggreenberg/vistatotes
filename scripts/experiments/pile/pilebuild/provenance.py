"""Which machine and which *code* produced a cell, and a hash that outlives it (#3160, #3693)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pile_config as pc

from pilebuild.env import cells_io, log


def _device_record(embed_batch_size: int | None = None) -> dict:
    """Everything about the build that a later reader needs to compare cells.

    ``gres/gpu:v100`` is a *type*, and #3143 measured that a type is not a
    device: two nodes both answering to it produced ``siglip2_l`` vectors 1.5e-04
    apart, while three other devices agreed to ~1e-12. Nothing in ``scontrol`` or
    ``--gres`` distinguishes the parts, so the only way a rebuild can be told
    apart from the cell it replaces is if the build **writes down** what it ran
    on. That is what this is; it does not make the arithmetic reproducible, it
    makes the difference visible.
    """
    import torch  # noqa: PLC0415

    from vtscore.config import EMBED_PRECISION, embed_precision  # noqa: PLC0415

    rec: dict = {
        "hostname": os.uname().nodename,
        # The host, not the card, turned out to be the axis (#3160): PyTorch's
        # CPU kernel dispatch decides how the 384px resize rounds, and an AVX2
        # host disagrees with an AVX-512 one on 12.3% of pixels by one 8-bit
        # level -- which is the whole of the 1.5e-04 that #3143 attributed to the
        # GPU. Pinning the dispatch removes it; recording it explains a cell that
        # was built before the pin.
        "cpu": _cpu_model(),
        # What the process RESOLVED, not what was asked for. `ATEN_CPU_CAPABILITY`
        # is advisory: an unsupported or misspelled value is ignored in silence,
        # so echoing the request would record a pin that never took. Both are
        # written; a reader can see a request that did not land.
        "cpu_capability": _cpu_capability(),
        "aten_cpu_capability_requested": os.environ.get("ATEN_CPU_CAPABILITY"),
        # Not a property of the machine, and here anyway for the same reason the
        # line above is: it is a build parameter that moves the vectors. A
        # per-image embedding is supposed to be independent of what it was
        # batched with, and #3683 measured that it is not -- rebuilding
        # `siglip2_l` at batch 31 instead of 32, same images and same node,
        # changed 27 of 7,746 vectors by up to 1.6e-04, because the batched
        # GEMM's reduction order is not. That is 400x the same-node floor and
        # larger than the fp16 difference #3143 rejected, and until this key
        # existed nothing in the sidecar said what a cell had been batched at.
        "embed_batch_size": embed_batch_size,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_gres": os.environ.get("SLURM_JOB_GRES") or os.environ.get("SBATCH_GRES"),
        "precision_requested": EMBED_PRECISION,
        "precision_resolved": embed_precision(),
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        # The node is not the only unrecorded axis. `requirements/image-embedders.txt`
        # pins `transformers>=4.49`, and v5 renamed the image processors: the plain
        # name is now the torchvision implementation and the PIL one moved to a
        # `Pil` suffix. So two hosts resolving different versions preprocess the
        # same image differently -- measured at 7.8e-3 max abs in pixels between
        # the two paths, well above the 1.5e-04 device effect this record was
        # written for. Recording the version and the class that actually loaded
        # costs nothing and makes that axis visible too.
        "transformers": _transformers_version(),
    }
    if not torch.cuda.is_available():
        rec["gpu_name"] = None
        rec["note"] = "no CUDA device; embedded on CPU"
        return rec
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    rec.update(
        {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": f"sm_{major}{minor}",
            # SM count was #3143's leading hypothesis for the drift (different
            # tiling, different accumulation order). It is not the cause -- the
            # two V100 parts have 80 SMs each and produce bit-identical GEMMs --
            # but it stays in the record because it is the cheapest way to tell
            # two cards apart that share a name.
            "multi_processor_count": props.multi_processor_count,
            "total_memory_gb": round(props.total_memory / 1e9, 1),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    )
    return rec


def effective_embed_batch_size(embedder: str) -> int | None:
    """The batch size *embedder* will forward at, given the environment right now.

    Read the embedder rather than the env var, because the env var is only the
    default: a subclass with a tighter VRAM budget passes its own smaller one to
    ``resolve_embed_batch_size``, and it is the number the GEMM sees that moves
    the vectors. Registry lookup only -- no weights load, so this is free to
    call before the pass it describes.

    **Call it while the build's ``VTSEARCH_EMBED_BATCH_SIZE`` is still set.**
    ``build_pile`` applies the per-embedder size for the duration of the embed
    pass and pops it afterwards, so asking at provenance-write time would answer
    with the shipped default -- recording a size the pass never ran at, which is
    the failure the ``aten_cpu_capability_requested`` comment above warns about.
    """
    try:
        from vtscore.media import get_embedder  # noqa: PLC0415

        return int(get_embedder(embedder).embed_batch_size)
    except Exception:  # noqa: BLE001 -- provenance must never fail a build
        return None


def _cpu_capability() -> str | None:
    """The CPU kernel ISA torch is actually dispatching to (``AVX512``/``AVX2``/...)."""
    import torch  # noqa: PLC0415

    getter = getattr(getattr(torch.backends, "cpu", None), "get_cpu_capability", None)
    return str(getter()) if getter else None


def _cpu_model() -> str | None:
    """The host CPU model string, or None where /proc/cpuinfo is not readable."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _transformers_version() -> str | None:
    try:
        import transformers  # noqa: PLC0415
    except ImportError:
        return None
    return getattr(transformers, "__version__", None)


def _processor_record(embedder: str) -> dict:
    """The preprocessing classes this embedder actually resolved to.

    Best effort: an embedder with no HF processor (or one that failed to load)
    records nulls rather than sinking a build that has already produced a cell.
    """
    try:
        from vtscore.media import get_embedder  # noqa: PLC0415

        emb = get_embedder(embedder)
        proc = getattr(emb, "_processor", None)
        image_proc = getattr(proc, "image_processor", None)
        return {
            "processor_class": type(proc).__name__ if proc is not None else None,
            "image_processor_class": type(image_proc).__name__ if image_proc is not None else None,
        }
    except Exception as exc:  # noqa: BLE001 -- provenance must never fail a build
        return {"processor_class": None, "image_processor_class": None, "error": repr(exc)[:120]}


def _git(repo: Path, *args: str) -> str | None:
    """One `git -C repo ...`, or None when git or the repo is not usable."""
    import subprocess  # noqa: PLC0415, S404 -- fixed argv, no shell

    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _code_record() -> dict:
    """The checkout that is about to embed: which tree, at which commit.

    The commit alone was here before #3693, and it was not enough. The launcher
    built the pile from a **fixed path** rather than its own location, so
    `bash launch_pile.sh vg_scale` from any other worktree submitted jobs that
    imported `/exp/$USER/projects/vts-pile` -- 1,420 commits behind `dev`,
    predating `vg_scale` entirely. A commit hash records *what* ran and cannot
    say *which tree it came from*, so a cell built by a stale checkout and a cell
    built by a current one were indistinguishable in the sidecar until someone
    resolved the hash by hand. The path is the field that makes the mix-up
    legible; `provenance_report` calls out a pile built from more than one.

    ``commit_at_launch`` is the launcher's own reading of HEAD, carried in by
    `launch_pile.sh`. A pile job queues for hours: a worktree that changes branch
    between submit and start builds from code the launch banner never showed
    anyone, and the two fields disagreeing is the only trace of it.
    """
    repo = Path(os.environ.get("VTS_REPO") or Path(__file__).resolve().parents[4])
    commit = _git(repo, "rev-parse", "HEAD")
    launched = os.environ.get("VTS_LAUNCH_COMMIT") or None
    return {
        "repo": str(repo),
        "commit": commit,
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        # Uncommitted tracked changes mean the commit above does not describe
        # what ran. Recorded rather than refused: a build is not the moment to
        # discover it, and a null (git unavailable) is not the same as clean.
        "dirty": (None if commit is None else bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))),
        "commit_at_launch": launched,
        "matches_launch": (None if not launched or not commit else launched == commit),
    }


def cell_fingerprint(dataset: str, embedder: str, medias: dict | None = None) -> dict:
    """A hash of the cell's vectors, in a fixed media-id order.

    The point of the hash is that it survives the cell it describes: a rebuild
    can be compared against it without keeping the old 900 MB pickle, which is
    exactly the check a purge-and-rebuild needs and cannot otherwise make.

    ``medias`` lets a fresh build hand over what it already has; the patch cells
    are 3.5 GB on disk, and re-reading one purely to hash it would add minutes
    and a second copy in RAM to every build.
    """
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    if medias is None:
        medias = cells_io().load_medias(pc.cell_path(dataset, embedder))
    ids = sorted(medias)
    vecs = [media_embedding(medias[i]) for i in ids]
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in vecs if v is not None])
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return {
        "n_vectors": int(arr.shape[0]),
        "dim": int(arr.shape[1]) if arr.ndim > 1 else None,
        "vectors_sha256": digest,
        "id_range": [int(ids[0]), int(ids[-1])] if ids else None,
    }


def write_provenance(
    dataset: str,
    embedder: str,
    summary: dict,
    medias: dict | None = None,
    embed_batch_size: int | None = None,
) -> Path:
    """Write the per-cell provenance sidecar.

    *embed_batch_size* is what the embed pass actually ran at; see
    :func:`effective_embed_batch_size` for why the caller has to measure it
    rather than this function reading the environment.
    """
    record = {
        "dataset": dataset,
        "embedder": embedder,
        "cell": pc.cell_path(dataset, embedder).name,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": _device_record(embed_batch_size),
        "code": _code_record(),
        "preprocessing": _processor_record(embedder),
        "cell_summary": {k: v for k, v in summary.items() if k != "status"},
        "fingerprint": cell_fingerprint(dataset, embedder, medias),
    }
    path = pc.provenance_path(dataset, embedder)
    path.write_text(json.dumps(record, indent=2) + "\n")
    dev = record["device"]
    code = record["code"]
    log(f"  provenance: {dev.get('gpu_name')} on {dev.get('hostname')} -> {path.name}")
    # Which code, said out loud in the build log too: the sidecar is only read
    # by someone who already suspects something (#3693).
    log(f"  built from: {code.get('repo')} @ {(code.get('commit') or 'unknown')[:9]} ({code.get('branch')})")
    if code.get("dirty"):
        log("  WARNING: that checkout had uncommitted tracked changes -- this cell is not reproducible")
    if code.get("matches_launch") is False:
        log(
            f"  WARNING: launched from {(code.get('commit_at_launch') or '')[:9]}, built at "
            f"{(code.get('commit') or '')[:9]} -- the checkout MOVED while this job was queued"
        )
    return path
