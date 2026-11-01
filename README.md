# AffineSym

Code for **Identifiability of Emotion-Conditioned Gating in Affine Fusion Architectures**.

Mayte H. Laureano, Hiram Calvo, José E. Valdez-Rodríguez, Pablo Gervás, and Gonzalo Méndez.

We study a class of affine fusion architectures in which complementary gate values can produce exactly the same logits after a corresponding parameter transformation. This shows why conditioning a gate on emotion does not, by itself, make its value an explanation of emotional influence.

Dual-path DistilBERT with NRC-VAD features provides the case study, comparing text-only classification, feature concatenation, and scalar modulation on a mental-health text benchmark.

## Contents

| File | Purpose |
|---|---|
| `scripts/run_experiments.py` | Run the pilot or full experiments. |
| `scripts/experiment_pipeline.py` | Prepare data, train models, and compute gate diagnostics. |
| `scripts/recompute_statistics.py` | Recalculate statistics from saved supplementary records. |
| `scripts/regenerate_tables_figures.py` | Recreate the manuscript tables and figures. |
| `scripts/audit_original_archive.py` | Check the original experiment archive. |

## Reproduction

Reproducing the analyses without retraining also requires the companion `supplementary_data/` directory.

Contact: [Hiram Calvo](mailto:hcalvo@cic.ipn.mx).
