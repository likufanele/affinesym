# AffineSym

Research code for **Identifiability of Emotion-Conditioned Gating in Affine Fusion Architectures**.

AffineSym examines whether the numerical value of an emotion-conditioned gate identifies how strongly affective information influences a prediction. The manuscript establishes a complement symmetry for a defined class of scalar-gated affine fusion architectures, then studies its practical implications using dual-path DistilBERT on a mental-health text classification benchmark.

The theoretical result is independent of the encoder. The experiments characterize predictive performance, gate interventions, and diagnostic associations in the DistilBERT case study.

## The architectural result

Consider an encoded representation $h$, conditioning features $v$, and a scalar gate $\alpha(v)$. The studied inference-time logits have the form

$$
z(h,v)=Ah+Cv+\alpha(v)(Bh+q)+\beta.
$$

If the controller can represent $1-\alpha$ and the affine parameters admit the following transformation,

$$
\alpha'=1-\alpha,\qquad A'=A+B,\qquad B'=-B,\qquad C'=C,
\qquad q'=-q,\qquad \beta'=\beta+q,
$$

then $z'(h,v)=z(h,v)$ for every input. Strict gate rankings reverse while all inference-time logits remain unchanged. A sigmoid controller with an unconstrained final affine layer admits the complementary gate by negating that layer's weights and bias.

The result requires the specified fusion structure, admissible parameter transformation, and deterministic evaluation. It does not automatically apply to every architecture described as gating, attention, or FiLM, and it does not imply identical training dynamics.

A centered intervention contrast is invariant under this complement transformation:

$$
\Delta z=z(\alpha)-z(0.5)=(\alpha-0.5)(Bh+q).
$$

This contrast measures the fitted gate's effect relative to a fixed midpoint. It preserves the direct conditioning path and is therefore not a measure of total affective influence. For this architectural class, conditioning a gate on emotion does not make its numerical value an explanation of how strongly emotion influences a prediction.

## Repository layout

The commands below assume that the Python files are in `scripts/`, with this README in the repository root.

| Path | Purpose |
|---|---|
| `scripts/run_experiments.py` | Command-line entry point for pilot or full training runs. |
| `scripts/experiment_pipeline.py` | Data preparation, duplicate grouping, VAD extraction, models, training, inference, interventions, and bootstrap comparisons. |
| `scripts/recompute_statistics.py` | Recompute scores, correlations, intervention summaries, and optional paired intervals from the compact supplementary records. |
| `scripts/regenerate_tables_figures.py` | Regenerate manuscript LaTeX tables and PDF figures from the supplementary records. |
| `scripts/audit_original_archive.py` | Audit the original full experiment archive, including records omitted from the compact supplement. |
| `supplementary_data/` | Companion data needed for analysis without retraining; copy this directory from the manuscript's reproducibility package if it is not included in your checkout. |

`Classifier.complemented_logits` implements the transformed fusion head. During diagnostic inference, the code compares its logits with the learned head and checks that the maximum absolute difference is below `1e-4`.

## Installation

For analysis of the saved supplementary records, install:

```bash
python -m pip install numpy pandas scipy matplotlib
```

Training additionally requires PyTorch, Transformers, scikit-learn, datasketch, huggingface_hub, and tqdm. Install a PyTorch build appropriate for your platform using the [official installation guide](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install transformers huggingface_hub scikit-learn datasketch tqdm
```

These commands install the dependencies without locking their versions. To reproduce the reported environment, use the versions recorded below and in `supplementary_data/run_configurations.json`. An NVIDIA GPU is appropriate for training; saved-record analysis requires neither a GPU nor PyTorch. The training code falls back to CPU if CUDA is unavailable.

### Recorded experimental environment

| Component | Recorded value |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| Transformers | 5.21.0 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.6.1 |
| datasketch | 1.6.5 |

These are the versions recorded by the experiments, not a compatibility claim for every platform. The wrapper pins the pretrained model revision to `12040accade4e8a0f71eabdb258fecc2e7e948be` for `distilbert/distilbert-base-uncased`. Matching inputs, software, and seeds improves reproducibility but does not guarantee bitwise-identical training across hardware.

## Input data

Obtain the [Sentiment Analysis for Mental Health corpus](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) and the unigram file from [NRC-VAD v2.1](https://saifmohammad.com/WebPages/nrc-vad.html) from their original providers. The corpus CSV must contain `statement` and `status` columns. The lexicon input is the headered, tab-separated unigram file with word, valence, arousal, and dominance columns.

The reported experiment uses 52,681 retained statements with 7 source labels. Detected exact and near-duplicate groups are kept within a single partition, giving 36,885 training, 7,899 validation, and 7,897 test records. Near-duplicate retrieval is approximate; these partitions are not claimed to be user-disjoint or source-disjoint.

Input and partition SHA-256 hashes are recorded in `supplementary_data/fingerprint.json` and in each new run's data directory. The source corpus and lexicon are not bundled with the scripts. Obtain and use them under their providers' terms; NRC-VAD's download page specifies research/educational non-commercial use.

## Run the case study

Run commands from the repository root. The paths below assume that the downloaded inputs have been placed in a local `data/` directory; adjust them to your filenames.

Start with the pilot configuration:

```bash
python scripts/run_experiments.py \
  --corpus "data/Combined Data.csv" \
  --lexicon "data/unigrams-NRC-VAD-Lexicon-v2.1.txt" \
  --output outputs/pilot \
  --mode pilot
```

The pilot trains 3 models at 32 subwords with seed 42: text only, full-document VAD concatenation, and full-document VAD modulation. It uses the full data preparation and 3-epoch training protocol; it is not a tiny-data smoke test.

Run the complete experiment grid with:

```bash
python scripts/run_experiments.py \
  --corpus "data/Combined Data.csv" \
  --lexicon "data/unigrams-NRC-VAD-Lexicon-v2.1.txt" \
  --output outputs/full \
  --mode full
```

The full grid contains 50 training runs: 2 input limits (32 and 128 subwords), 5 model/scope configurations, and 5 seeds (7, 13, 21, 42, and 73).

| Model | VAD scope | Classifier input |
|---|---|---|
| Text only | None | $h$ |
| Concatenation | Full document | $[h;v]$ |
| Concatenation | Visible span | $[h;v]$ |
| Scalar modulation | Full document | $[h;\alpha h;v;\alpha]$ |
| Scalar modulation | Visible span | $[h;\alpha h;v;\alpha]$ |

The VAD vector has 8 coordinates: means and standard deviations of valence, arousal, and dominance, lexical coverage, and `log(1 + match_count)`. Visible-span features include only complete lexical units visible within the encoder's input window. Feature scaling is fitted on the training partition.

The wrapper sets 3 epochs, AdamW with learning rate `2e-5` and weight decay `0.01`, class-weighted cross-entropy, microbatches of 16, and an effective batch size of 512. Selection uses validation Macro-F1. The full summary computes 105 paired comparisons with 3,000 test-group bootstrap resamples each. These settings are defined in `run_experiments.py`; they are not additional command-line flags.

Completed runs with matching configuration records are reused. An interrupted run without `done.json` is retrained rather than resumed from an optimizer checkpoint.

### Generated outputs

The output directory contains `data_<hash>/` for preparation records, `runs/` for individual fitted models, and `summaries/<hash>/` for aggregate scores and comparisons. Run directories include `config.json`, `history.csv`, `metrics.json`, `predictions.csv.gz`, `logits.npz`, and `best.pt`. The supplied wrapper retains checkpoints for every model.

Modulation runs also write `mentalhealth_scalar_gate_instance_analysis.csv`, containing gate values, fixed-gate predictions, logit distances, complement residuals, and source statements. Use the compact text-free supplementary records when distributing the paper's numerical analyses. The original full result archive and trained checkpoints are not part of that compact supplement.

Fresh training outputs and the compact `supplementary_data/` release have different layouts. The training entry point does not automatically assemble the complete compact release expected by the two manuscript-analysis scripts below.

## Reproduce analyses without retraining

Place the complete companion `supplementary_data/` directory beside `scripts/`. It includes the split manifest, all 50 prediction sets, the 20 modulation diagnostic sets, reference summaries, and the declared qualitative selections. Full source statements are omitted. Row IDs refer to positions in the original corpus CSV before missing-value removal.

Recompute and check the saved metrics and diagnostics:

```bash
python scripts/recompute_statistics.py \
  --data supplementary_data \
  --output outputs/recomputed
```

To also repeat all 105 paired group-bootstrap intervals:

```bash
python scripts/recompute_statistics.py \
  --data supplementary_data \
  --output outputs/recomputed \
  --bootstrap
```

The script checks the recalculated scores, correlations, and requested intervals against the supplied reference records. Bootstrap intervals describe test-sample uncertainty conditional on each fitted model pair; training-seed variability is reported separately.

Create the output folders and regenerate the manuscript tables and figures:

```bash
python -c "from pathlib import Path; Path('tables').mkdir(exist_ok=True); Path('figures').mkdir(exist_ok=True)"
python scripts/regenerate_tables_figures.py
```

The plotting script reads `supplementary_data/` relative to the repository root and writes LaTeX tables to `tables/` and PDF figures to `figures/`, replacing files of the same names. It uses the supplied summary records, not the newly written `outputs/recomputed/` directory.

For an optional audit of the original full experiment archive:

```bash
python scripts/audit_original_archive.py \
  "resultados_full_4cdc1d27e5ce.zip" \
  outputs/archive_audit
```

This command expects the full experiment archive, not the manuscript source ZIP. An original corpus CSV can be supplied as a third positional argument for the additional corpus checks.

## Case-study findings

Increasing the input limit from 32 to 128 subwords raises text-only mean Macro-F1 from **0.6537 to 0.7297**. Scalar modulation shows no consistent advantage over matched concatenation in the paired group-bootstrap comparisons.

Replacing the learned gate by 0.5 while keeping trained parameters and representations fixed changes **2–26 of 7,897 predictions** per modulation run. The maximum numerical complement residual is approximately **2.38 × 10⁻⁶**, consistent with floating-point evaluation of the exact algebraic identity. This residual is numerical error, not the strength of the symmetry.

Gate correlations vary across seeds and after adjustment for text length and source label. The illustrative qualitative analysis uses 128 subwords, visible-span VAD, and seed 42. Its 20 cases are selected using within-class gate extrema followed by high-entropy records; they are not a random sample.

These empirical findings concern one corpus, encoder, lexicon, grouped partition, and fixed training budget. The task reproduces source-provided benchmark labels; it is not clinical diagnosis. Document-level VAD summaries do not model temporal emotional trajectories.

## Associated manuscript

Mayte H. Laureano, Hiram Calvo, José E. Valdez-Rodríguez, Pablo Gervás, and Gonzalo Méndez. *Identifiability of Emotion-Conditioned Gating in Affine Fusion Architectures*. Manuscript, 2026.

```bibtex
@misc{laureano2026affinesym,
  author = {Laureano, Mayte H. and Calvo, Hiram and
            Valdez-Rodr{\'i}guez, Jos{\'e} E. and
            Gerv{\'a}s, Pablo and M{\'e}ndez, Gonzalo},
  title = {Identifiability of Emotion-Conditioned Gating
           in Affine Fusion Architectures},
  year = {2026},
  note = {Manuscript}
}
```

Correspondence: [Hiram Calvo](mailto:hcalvo@cic.ipn.mx).
