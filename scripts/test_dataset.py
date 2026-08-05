#!/usr/bin/env python
"""Test CorrectionDataset."""

from cellmap_flow.finetune.dataset import CorrectionDataset, create_dataloader

def main():
    print("="*60)
    print("Testing CorrectionDataset")
    print("="*60)

    # Test dataset
    print("\n1. Creating dataset...")
    dataset = CorrectionDataset(
        "test_corrections.zarr",
        patch_shape=(64, 64, 64),
        augment=True,
        normalize=True,
    )
    print(f"✓ Dataset loaded: {len(dataset)} corrections")

    # Test loading a sample
    print("\n2. Loading first sample...")
    raw, target = dataset[0]
    print(f"  Raw shape: {raw.shape}, dtype: {raw.dtype}")
    print(f"  Raw range: [{raw.min():.3f}, {raw.max():.3f}]")
    print(f"  Target shape: {target.shape}, dtype: {target.dtype}")
    print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")

    # Test augmentation consistency
    print("\n3. Testing augmentation...")
    raw1, _ = dataset[0]
    raw2, _ = dataset[0]
    print(f"  Sample 1 range: [{raw1.min():.3f}, {raw1.max():.3f}]")
    print(f"  Sample 2 range: [{raw2.min():.3f}, {raw2.max():.3f}]")
    if not (raw1 == raw2).all():
        print("  ✓ Augmentation working (samples differ)")
    else:
        print("  ! Warning: Samples identical (augmentation may not be working)")

    # Test DataLoader
    print("\n4. Creating DataLoader...")
    dataloader = create_dataloader(
        "test_corrections.zarr",
        batch_size=2,
        patch_shape=(64, 64, 64),
        num_workers=2,
        shuffle=True,
    )
    print(f"✓ DataLoader created: {len(dataloader)} batches")

    # Test batch loading
    print("\n5. Loading first batch...")
    for raw_batch, target_batch in dataloader:
        print(f"  Raw batch shape: {raw_batch.shape}")
        print(f"  Target batch shape: {target_batch.shape}")
        print(f"  Raw batch range: [{raw_batch.min():.3f}, {raw_batch.max():.3f}]")
        print(f"  Target batch range: [{target_batch.min():.3f}, {target_batch.max():.3f}]")
        break

    # Test memory usage
    print("\n6. Testing multiple batches...")
    batch_count = 0
    for raw_batch, target_batch in dataloader:
        batch_count += 1
        if batch_count >= 3:
            break
    print(f"✓ Successfully loaded {batch_count} batches")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    main()
