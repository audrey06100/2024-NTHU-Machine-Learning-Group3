import importlib
import time
import os

import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])



def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(sr, clean_signal, noisy_signal, "wb")


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    frames_total = len(data_a)
    if frames_total < sample_length:
      new_data_a = np.zeros((sample_length,), dtype=data_a.dtype)
      new_data_b = np.zeros((sample_length,), dtype=data_b.dtype)
      
      new_data_a[:frames_total] = data_a
      new_data_b[:frames_total] = data_b
      data_a, data_b = new_data_a, new_data_b
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."


    #start = np.random.randint(frames_total - sample_length + 1)
    min_start_time = 0.5*16000
    max_start_time = max(1, frames_total-sample_length-0.5*16000)
    if min_start_time > max_start_time:
      min_start_time = 0
    start_idx = np.random.randint(min_start_time, max_start_time)
    #print(f"Random crop from: {start_idx}")
    
    end_idx = start_idx + sample_length

    return data_a[start_idx:end_idx], data_b[start_idx:end_idx]
    '''
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"

    frames_total = len(data_a)

    start = 0 if frames_total<(1*16000+sample_length) else 1*16000
    end = start + sample_length

    if frames_total < sample_length:
      new_data_a = np.zeros((sample_length,), dtype=data_a.dtype)
      new_data_b = np.zeros((sample_length,), dtype=data_b.dtype)
      
      new_data_a[:frames_total] = data_a
      new_data_b[:frames_total] = data_b
      data_a, data_b = new_data_a, new_data_b
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    return data_a[start:end], data_b[start:end]
    '''


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")
