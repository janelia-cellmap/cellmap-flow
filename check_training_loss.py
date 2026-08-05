#!/usr/bin/env python3
"""Check training loss from checkpoint file."""

import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check training loss from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., output/fly_organelles_mito_liver/best_checkpoint.pth)"
    )

    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)

    # Print basic info
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")
    if 'best_loss' in checkpoint:
        print(f"Best loss: {checkpoint['best_loss']:.6f}")

    # Print training stats history if available
    if 'training_stats' in checkpoint and checkpoint['training_stats']:
        print("\n" + "="*60)
        print("LOSS HISTORY")
        print("="*60)
        print(f"{'Epoch':<10} {'Loss':<15} {'Best Loss':<15}")
        print("-"*60)

        for stat in checkpoint['training_stats']:
            epoch = stat.get('epoch', '-')
            loss = stat.get('loss', float('nan'))
            best_loss = stat.get('best_loss', float('nan'))
            print(f"{epoch:<10} {loss:<15.6f} {best_loss:<15.6f}")

        # Check if loss decreased
        losses = [stat['loss'] for stat in checkpoint['training_stats']]
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Initial loss:  {initial_loss:.6f}")
        print(f"Final loss:    {final_loss:.6f}")
        print(f"Improvement:   {improvement:.2f}%")

        if improvement < 1:
            print("\n⚠ WARNING: Loss barely improved (<1%)!")
            print("  → Training may not be working properly")
            print("  → Check data quality with analyze_corrections.py")
        elif improvement < 10:
            print("\n⚠ MODERATE: Loss improved but not dramatically")
            print("  → Consider training longer or adjusting hyperparameters")
        else:
            print("\n✓ GOOD: Significant improvement in loss")

    else:
        print("\nNo training stats history found in checkpoint")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
