#!/usr/bin/env python
"""CLI entry point for voice dataset augmentation."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from augmentation.pipeline import AugmentationPipeline, PipelineConfig
from augmentation.disfluency import DisfluencyConfig
from augmentation.constants import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice dataset augmentation pipeline"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help="Comma-separated list of datasets to process",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets"),
        help="Base directory containing datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("augmented_data"),
        help="Output directory for augmented data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for LLM calls",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit samples per dataset (for testing)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="Comma-separated list of splits to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name to use for emotion tagging and other LLM tasks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    datasets = [d.strip() for d in args.datasets.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    
    print(f"Voice Dataset Augmentation Pipeline")
    print(f"=" * 40)
    print(f"Datasets: {datasets}")
    print(f"Splits: {splits}")
    print(f"Output: {args.output_dir}")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(datasets)} datasets")
        return
    
    config = PipelineConfig(
        model=args.model,
        datasets=datasets,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        disfluency_config=DisfluencyConfig(),
    )
    
    pipeline = AugmentationPipeline(config)
    
    print(f"\nProcessing...")
    stats = pipeline.run(splits=splits)
    
    print(f"\n{'Dataset':<15} {'Count':>10}")
    print("-" * 25)
    for dataset, count in stats.items():
        print(f"{dataset:<15} {count:>10}")
    
    total = sum(stats.values())
    print("-" * 25)
    print(f"{'Total':<15} {total:>10}")
    
    print(f"\nDone! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
