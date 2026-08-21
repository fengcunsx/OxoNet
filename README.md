# OxoNet

Code, trained weights and reproduction manifests for **"OxoNet: A Hybrid Deep Learning
Network for High-Specificity Detection of 8-oxo-dG from Nanopore Signals, with Evaluation
on Genomic Negatives and Native Human DNA"** (IEEE Access, manuscript Access-2026-24892).

> The tag `v1.3-resubmission` is the immutable snapshot corresponding to the submitted
> manuscript. `main` may continue to receive documentation and bug fixes.
>
> Earlier tags are kept for provenance but should not be used. `v1.0` and `v1.1` instantiate
> the inference entry points with a 5-base sequence input, which was never trained, and are
> missing the `dataset` package. `v1.2` fixes the architecture but leaves the entry points
> themselves broken: the model was never moved to the compute device, the CPU branch built an
> invalid `torch.device("")`, and the 7-mer input was silently trimmed to its central 5-mer.
> Every inference path now goes through `model/loader.py`, and `smoke_test.py` exercises it.

## What is here

| Path | Contents |
|---|---|
| `model/` | OxoNet architecture (Sig-Net, Seq-Net, cross-attention fusion) |
| `scripts/` | Training, epoch selection, prediction (all inference paths go through `model/loader.py`) |
| `scripts/analysis/` | Every script that produces a table or figure in the paper |
| `scripts/nanocon_bench/` | NanoCon baseline: scoring and CSV conversion |
| `scripts/train_curves/` | Per-epoch logs: OxoNet seeds 42/0/3407 and all five ablation arms (`*.tsv`), plus the three NanoCon-7 runs as TensorBoard event files |
| `manifests/` | Split manifests and the exact evaluated sites (see below) |
| `weights/` | `oxonet_seed42_ep125.pth` — the model used throughout the paper (seed 42, epoch 125, selected on the validation set by recall under a strict-specificity constraint). `nanocon_baseline_ep29.ckpt` — our retrained NanoCon baseline |
| `predictions/` | Scored probabilities of every seed and every ablation arm on both test sets |

## Reproduction manifests

`manifests/sites_{valid,test_oligo,test_t2t}.csv.gz` list every evaluated site as
`(read_id, basecall_pos, label)`; `manifests/reads_*.txt.gz` give the same read IDs as sorted
one-per-line lists.

**Split evidence and direct intersection checks.** Run `python verify_read_disjoint.py` (CPU,
seconds). It reports four layers of evidence separately rather than collapsing them, because they
are not equally direct, and its closing lines are deliberately not a single global PASS:

- *manifest* — the released assignment is well formed: seed 42, every entry labelled
  train/valid/test, group counts 846 = 762/42/42 (oligo) and 315 = 283/16/16 (genomic). This
  validates the manifest; it is not a read-identifier check.

- *exact* — the training positives are enumerated (`reads_train_pos.txt.gz`, 454,984 reads,
  taken from the array that was fed to the model). Their intersection with validation,
  test-oligo and test-genomic is 0, 0, 0.
- *by construction* — the training **negatives cannot be enumerated**: the packed negative
  training arrays kept only signal and summary features, and the intermediate per-read files
  were deleted after packing. Their disjointness follows instead from `split_manifest.json`
  and `dataset/build_split.py`: whole FAST5 directories and whole npz parts are assigned to one
  split *before any read is opened*, and a read lives in exactly one such file. This covers
  every read, including unarchived ones — which enumeration never could.
- *independent* — a later 13-mer extraction over the same partition did keep read IDs and also
  gives 0, 0, 0. It is corroboration, not provenance: it drops 540 reads the 7-mer build kept
  and adds 420 it did not, because the two extractions apply different gap filters. It needs
  the 43 GB 13-mer archive, which is not redistributed, and is skipped when absent.

On a fresh clone the 13-mer archive is absent, so what runs unaided is the manifest validation and
the direct intersection over the training positives.

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
architecture, produces a forward pass, and does so through `model.loader.build_model` — the code
path `scripts/` actually uses. Also CPU-only.

Run `python verify_read_disjoint.py` for the split audit described above.

A release is considered acceptable only when all four of these pass in a fresh clone:

```bash
python -m compileall -q .   # every file parses
python smoke_test.py        # checkpoint + entry point
python verify_read_disjoint.py
python reproduce_tables.py  # numbers match the paper
```

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
from model.loader import build_model

model, device = build_model('weights/oxonet_seed42_ep125.pth')
```

`build_model` is the only supported way in: it fixes the architecture at `seq_l=7` with RoPE,
loads with `strict=True`, moves the model to a valid device and calls `eval()`. It accepts the
three checkpoint layouts this project has produced (bare `state_dict`, `{'model': ...}`,
`{'model_state_dict': ...}`); the released weights are the first.

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
