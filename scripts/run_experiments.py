"""Command-line entry point for the experiment pipeline supplied with the revision."""
import argparse
from pathlib import Path
from experiment_pipeline import prepare_data, experiment_grid, run_one, summarize

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--lexicon', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--mode', choices=['pilot', 'full'], default='full')
    args = p.parse_args()
    cfg = dict(data_path=str(args.corpus), lexicon_path=str(args.lexicon),
               output_dir=str(args.output), split_seed=42, near_duplicates=True,
               model_name='distilbert/distilbert-base-uncased',
               model_revision='12040accade4e8a0f71eabdb258fecc2e7e948be',
               epochs=3, lr=2e-5, micro_batch=16, effective_batch=512,
               eval_batch=32, keep_weights='all',
               bootstrap_samples=3000)
    data = prepare_data(cfg)
    folders = [run_one(data, cfg, *spec) for spec in experiment_grid(args.mode)]
    summarize(data, cfg, folders)

if __name__ == '__main__':
    main()
