import logging

import torch


def check_ampere_gpu():
    """Check if the GPU supports NVIDIA Ampere or later and enable FP32 in PyTorch if it does."""

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.info("No GPU detected, running on CPU.")
        return False

    gpu_name = torch.cuda.get_device_name()

    # Supported GPU models list - roughly based on https://en.wikipedia.org/wiki/Ampere_(microarchitecture)
    supported_gpus = ["A100", "A6000", "RTX 30", "RTX 40", "A30", "A40", "H100", "B200"]

    # Checking all GPUs
    return any(supported_gpu in gpu_name for supported_gpu in supported_gpus)


def check_hpu():
    try:
        import habana_frameworks.torch.hpu as hthpu  # pylint: disable=import-outside-toplevel

        return hthpu.is_available()
    except ModuleNotFoundError:
        return False


def supports_bf16():
    return check_hpu() or check_ampere_gpu()


def recursive_map(item, fn):
    if isinstance(item, dict):
        return {k: recursive_map(v, fn) for k, v in item.items()}
    if isinstance(item, list):
        return [recursive_map(v, fn) for v in item]
    if torch.is_tensor(item):
        return fn(item)
    return item
