# OxoNet

Code, trained weights and reproduction manifests for **"OxoNet: A Hybrid Deep Learning
Network for High-Specificity Detection of 8-oxo-dG from Nanopore Signals, with Evaluation
on Genomic Negatives and Native Human DNA"** (IEEE Access, manuscript Access-2026-24892).

> The tag `v1.2-resubmission` is the immutable snapshot corresponding to the submitted
> manuscript. `main` may continue to receive documentation and bug fixes.
> (`v1.0` and `v1.1` are earlier snapshots and contain known defects: inference entry points
> instantiated with a 5-base sequence input, and a missing dataset package.)

## What is here

| Path | Contents |
|---|---|
| `model/` | OxoNet architecture (Sig-Net, Seq-Net, cross-attention fusion) |
| `scripts/` | Training, epoch selection, prediction |
| `scripts/analysis/` | Every script that produces a table or figure in the paper |
| `scripts/nanocon_bench/` | NanoCon baseline: scoring and CSV conversion |
| `scripts/train_curves/` | Per-epoch training/validation logs for all seeds and ablation arms |
| `manifests/` | Split manifests and the exact evaluated sites (see below) |
| `weights/` | `oxonet_seed42_ep125.pth` — the model used throughout the paper (seed 42, epoch 125, selected on the validation set by recall under a strict-specificity constraint). `nanocon_baseline_ep29.ckpt` — our retrained NanoCon baseline |
| `predictions/` | Scored probabilities of every seed and every ablation arm on both test sets |

## Reproduction manifests

`manifests/sites_{valid,test_oligo,test_t2t}.csv.gz` list every evaluated site as
`(read_id, basecall_pos, label)`. These define the split exactly: the read-id sets of the
three subsets are pairwise disjoint, which is the property the paper's leakage check verifies.

- `valid_genomic_neg_mask.npy` — a **heuristically inferred** genomic-enriched validation mask. The
  packed validation data does not record the origin of each negative, so a negative is treated as
  genomic when its read contributes more than 100 candidate sites (`scripts/analysis/valid_neg_source.py`).
  Varying that cut from 100 to 1000 moves recall at FPR 1e-4 by +1.07 points for OxoNet, +0.55 for
  NanoCon-7, +0.29 for esox and +0.26 for the tree control.
  **All primary comparative thresholds are fixed on this subset**, never on test data; the paper
  additionally reports one exploratory native-calibrated point, which is labelled as such.
- `valid_thresholds.json` — the thresholds and validation recalls per arm and operating point.
- `pos_train_ctx.json` — the 100 guanine-centred 5-mer contexts that delimit which sites are
  interrogated at all.

## Environment

`requirements.txt` records the environment that produced the released weights (Python 3.11,
PyTorch 2.1.2+cu121). The two baselines run in their own environments with mutually incompatible
PyTorch versions; the file lists those too, together with the upstream repositories.

## Reproducing the ablation and reproducibility tables without a GPU

Run `python reproduce_tables.py`. It reads only `predictions/` and `manifests/` and prints the
recall columns of Tables 7 and 8 of the paper. No GPU, no weights and no raw data are needed.

Run `python smoke_test.py` to check that the released checkpoint loads strictly into the 7-mer
architecture and produces a forward pass. Also CPU-only.

`scripts/analysis/make_figures.py` regenerates the figures, but it reads scoring outputs from paths
on the machine where the analyses were run; it is released for inspection of how each figure was
produced rather than as a turnkey pipeline.

`predictions/` holds the scored probability of every arm on `test_oligo` and `test_t2t`, aligned
row-for-row with `manifests/sites_test_*.csv.gz`. Combined with `manifests/valid_thresholds.json`
this is enough to recompute Tables 7 and 8 exactly — no weights, no GPU, no re-training:

```python
import numpy as np, json
th = json.load(open('manifests/valid_thresholds.json'))
lab = np.loadtxt('manifests/sites_test_t2t.csv.gz', delimiter=',', skiprows=1, usecols=2)
p   = np.load('predictions/probs_full_seed42_ep125_test_t2t.npy')
recall = (p[lab == 1] >= th['full_seed42_ep125']['T@1e-04']).mean()   # -> 0.5590
```

We release these rather than the weights of the five ablation arms: the arms exist to justify
numbers in the paper, and the predictions let anyone verify those numbers directly, which the
weights alone would not.

Not everything in the paper can be recomputed from this repository alone. The tables that involve
esox or NanoCon need those methods' own environments and released models; the native analyses need
the raw FAST5 data and the full basecalling and resquiggling pipeline. What is self-contained here
is the OxoNet side: architecture, the checkpoint used throughout, the exact evaluated sites, the
thresholds, and every arm's predictions.

## Operating-point convention

Results are reported at a **matched false-positive rate on held-out genomic negatives**
(primary: 1e-4), not at a probability threshold. `FPR = 1e-6` is never reported: the negative
set is too small to resolve it. See Section IV-F of the paper.

## Loading the model

```python
from model.model import DetectModel
import torch

m = DetectModel(dim=128, sig_blocks=4, sig_l=175, seq_l=7, pos_mode='rope').eval()
m.load_state_dict(torch.load('weights/oxonet_seed42_ep125.pth', map_location='cpu'))
```

## Note on sequence width

Seq-Net consumes the full **7-mer** (`seq_l=7`); the central 5-mer is used only to define the
100 interrogated contexts. Instantiating `DetectModel(..., seq_l=5)` gives a different, smaller
model that was never trained.

## Data

Sequencing data are third-party and not redistributed here. Accessions are listed in the
paper's Data Availability Statement (ENA `PRJEB76712` for the synthetic and native runs;
the NA12878 genomic negatives are from the Nanopore WGS Consortium release 6).

## License

MIT (see `LICENSE`). If you use this work, please cite the paper — see `CITATION.cff`.
