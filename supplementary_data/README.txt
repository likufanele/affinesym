SUPPORTING DATA FOR applsci-4641089

These files document 50 new runs on a common grouped partition: 32/128 tokens,
text-only, concatenation or modulation, full/visible VAD where applicable,
and seeds 7, 13, 21, 42, 73. They replace the historical results; they are not
a reconstruction of the old three-seed experiment.

Row IDs are zero-based positions in the original Combined Data CSV, before
removing missing texts. Labels y_true/y_pred follow the ordered labels array
in audit.json. The original corpus and NRC-VAD lexicon are not redistributed.
The shared input and model hashes are in fingerprint.json.

split_manifest.csv identifies source labels, original/new split membership,
and detected duplicate groups. audit.json records overlap counts and detection
limits; near_duplicate_pairs.csv contains verified links. The grouping is not
a claim of source- or user-disjoint evaluation. class_counts.csv and the
truncation files provide the partition and input-length audit.

predictions_all_runs.csv.gz contains all 50 test prediction sets. Each row has
row_id, y_true, y_pred, length, variant, scope, and seed. The shared manifest
provides group_id. metrics_per_seed.csv and metrics_summary.csv are the exported
run scores; metrics_verified.csv and metrics_summary_verified.csv are audit
outputs. training_history.csv records all three epochs for each run, and
run_configurations.json contains exact model, training and environment settings.
Paths in exported metadata refer to the original Colab workspace and are not
required to read the included data.

gate_instances_all_runs.csv.gz contains the 20 modulation runs with raw and
standardized VAD features, full subword length, alpha, confidence, entropy,
probability margin, correctness, fixed-gate predicted labels, logit distances,
and the complement-check residual. Full source text is omitted. For learned
labels and predictions, join predictions_all_runs.csv.gz on row_id, length,
variant, scope, seed. The numerical complement residual is floating-point error
when checking an exact identity, not its effect size.

gate_intervention_per_seed.csv and gate_intervention_summary.csv report fixed
parameter comparisons against alpha=0.5. Positive delta_f1 means learned minus
fixed-gate F1. gate_correlations_per_seed.csv and its summary report raw and
partial rank correlations, controlling ranked full length and source labels.
gate_by_class_all_runs.csv retains the class-level diagnostics of all 20 runs.
length_matched_instances_all_runs.csv.gz contains the retained matching rows
and bin memberships for every modulation run. These subsets contain 602 rows
each; coarse matching does not cover the full length distributions.

paired_bootstrap_per_seed.csv contains 105 paired, within-seed comparisons,
with 3,000 shared group-bootstrap resamples and seed 20261031. Differences are
model_b minus model_a. The intervals are conditional on fitted model pairs;
they are not intervals over training seeds or familywise-adjusted tests.
paired_bootstrap_verified.csv and paired_contrasts_summary.csv add audit and
descriptive summaries. Interval endpoints must not be averaged across seeds.

The declared illustrative run is length=128, variant=gate, scope=visible,
seed=42. illustrative_cases.csv, illustrative_class_summary.csv and
illustrative_matched_class_summary.csv concern that run. The 20 cases are
selected by within-class alpha extrema, then highest remaining entropy, not
randomly. qualitative_cases_all_runs.csv retains the 400 selected records
across all modulation runs, with source text omitted. Distinct row IDs may
contain repeated text within the same held-out partition.

From the project directory, regenerate tables and figures with:
python scripts/regenerate_tables_figures.py

Recompute scores, correlations and intervention summaries with:
python scripts/recompute_statistics.py --output recomputed

Repeat all 105 conditional intervals as well with:
python scripts/recompute_statistics.py --output recomputed --bootstrap

These commands require numpy, pandas, scipy and matplotlib, and no GPU. The
recorded experimental versions are included in run_configurations.json.
The supplied figures and tables have already been generated.

To repeat training, obtain the corpus and NRC-VAD unigram v2.1 file from the
providers cited in the manuscript, verify their hashes, and use:
python scripts/run_experiments.py --corpus CORPUS.csv --lexicon LEXICON.txt --output NEW_RUNS

This command defaults to all 50 runs. It requires the recorded PyTorch,
Transformers, scikit-learn, datasketch, huggingface_hub, scipy, numpy, pandas,
and tqdm environment and a suitable GPU for comparable timing. The wrapper
pins the pretrained model revision. The core pipeline was supplied with the
new experiments; the only reproducibility modification is accepting an
explicit model revision instead of always resolving the current Hub revision.
The originally recorded CUDA/Python stack is not installed by these scripts.

audit_original_archive.py optionally audits the full original result ZIP,
including logits and scaling records that are not duplicated in this compact
supplement. Its usage is documented at the top of the script. The compact
recompute_statistics.py is sufficient for the reported score/correlation
tables and paired comparisons. Trained weights and full fixed-gate logits
are not included; repeating inference requires rerunning the training code.
