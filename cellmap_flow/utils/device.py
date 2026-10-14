"""The device torch work runs on, wherever cellmap-flow picks one."""


def best_device():
    """CUDA when there is one, else Apple's GPU (MPS), else the CPU.

    Every choice here was CUDA or the CPU, so a Mac served and trained its
    models on the CPU, though PyTorch runs on its GPU and Cellpose and
    bioimageio.core already used it on their own: a Cellpose finetune took
    about 33 s a batch of eight tiles there, two days for 20 epochs.
    """
    # Imported here: importing torch costs seconds, and the CLIs import the
    # modules that call this to build their command lists.
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
