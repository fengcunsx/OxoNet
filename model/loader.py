"""Single place where the released checkpoint is turned into a runnable model.

Every inference entry point in ``scripts/`` goes through :func:`build_model`.
Two things used to be got wrong independently in each script, so they are fixed
here once:

* the architecture is ``seq_l=7`` with RoPE.  ``seq_l=5`` was never trained and
  does not match any released checkpoint -- loading it raises rather than
  silently reshaping, because :func:`load_weights` uses ``strict=True``.
* checkpoints exist in three shapes depending on which script wrote them: a
  bare ``state_dict``, ``{'model': ...}`` (Lightning ``last.ckpt``) and
  ``{'model_state_dict': ...}`` (the older training script).  The released
  ``weights/oxonet_seed42_ep125.pth`` is the first of the three.

``KMER`` is the sequence width the model expects.  Dataset helpers that trim a
stored k-mer to a narrower window must be passed this value, not a literal.
"""
import torch

from model.model import DetectModel

KMER = 7


def pick_device(prefer_cuda=True):
    """A valid ``torch.device`` in both branches.

    ``torch.device("")`` is not a device and raises; it was the CPU branch of
    the original scripts, so they only ever ran on a CUDA host.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_weights(path, device):
    """Return a plain ``state_dict`` from any of the three checkpoint shapes."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def build_model(resume=None, device=None, prefer_cuda=True):
    """Instantiate the published architecture, optionally load weights, ``eval()``.

    Loading is strict: a checkpoint that does not match the architecture is an
    error, not something to be papered over with ``strict=False``.
    """
    device = pick_device(prefer_cuda) if device is None else device
    model = DetectModel(dim=128, sig_blocks=4, sig_l=175, seq_l=KMER,
                        pos_mode="rope").to(device)
    if resume is not None:
        model.load_state_dict(load_weights(resume, device), strict=True)
    model.eval()
    return model, device
