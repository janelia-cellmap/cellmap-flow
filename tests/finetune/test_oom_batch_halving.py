"""Halving the batch after an OOM keeps the loader as it was otherwise.

The OOM fallback rebuilt its DataLoader with only batch size, workers,
pin_memory and the multiprocessing context. Dropping persistent_workers
meant workers were re-spawned every epoch from a fresh copy of the dataset,
RNG unset and seed unchanged, so after an OOM every epoch drew the identical
patches and augmentations (and paid the worker start-up each time).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner


def _dataset():
    return TensorDataset(torch.rand(8, 1, 4, 4, 4), torch.ones(8, 1, 4, 4, 4))


def test_the_training_loader_keeps_its_workers_between_epochs():
    from cellmap_flow.finetune.virtual_dataset import make_training_loader

    loader = make_training_loader(_dataset(), batch_size=4, num_workers=2)
    assert loader.persistent_workers
    assert loader.multiprocessing_context is not None


def test_halving_keeps_workers_persistent(tmp_path):
    # What create_dataloader builds.
    loader = DataLoader(
        _dataset(), batch_size=4, shuffle=False, num_workers=2, pin_memory=True,
        persistent_workers=True, multiprocessing_context="spawn",
    )
    trainer = LoRAFinetuner(
        torch.nn.Conv3d(1, 1, 1), loader, output_dir=str(tmp_path), num_epochs=1,
        device="cpu", use_mixed_precision=False, loss_type="bce", tensorboard=False,
    )

    trainer._log_message = lambda message: None  # set by train() in a real run
    assert trainer._halve_batch_size()

    halved = trainer.dataloader
    assert halved.batch_size == 2
    assert trainer.gradient_accumulation_steps == 2
    assert halved.persistent_workers
    assert halved.num_workers == 2
    assert halved.pin_memory == loader.pin_memory
    assert halved.multiprocessing_context is loader.multiprocessing_context
    assert halved.dataset is loader.dataset


def test_rebuilding_keeps_the_sampling_order_kind():
    from cellmap_flow.finetune.virtual_dataset import rebuild_loader

    shuffled = DataLoader(_dataset(), batch_size=4, shuffle=True)
    in_order = DataLoader(_dataset(), batch_size=4, shuffle=False)
    assert isinstance(rebuild_loader(shuffled, 2).sampler, torch.utils.data.RandomSampler)
    assert isinstance(rebuild_loader(in_order, 2).sampler, torch.utils.data.SequentialSampler)
