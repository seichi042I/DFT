from typing import Tuple

import numpy as np
from tqdm import tqdm


def _dft(
    data: np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    """culcurate dft

    Args:
        data (np.ndarray): single channel signal data.
        sampling_rate (int): sampling rage
    Return:
        complex
    """

    f_range = len(data)//2 if len(data) < sampling_rate else sampling_rate//2
    n_dft = len(data)
    cfc = []  # complex fourier coefficient

    # 整数以外の周波数ωを使うとギザギザができるため回避
    quantize = (n_dft/sampling_rate) if (n_dft/sampling_rate) >= 1 else 1
    for f in range(f_range):
        omega = 2 * np.pi * f * quantize
        def e(n): return np.exp(-1j*(omega*n/n_dft))
        sigma = 0j
        for n, x in enumerate(data):
            sigma += x*e(n)
        cfc.append(2*sigma/n_dft)

    return np.array(cfc)


def dft(
    data: np.ndarray,
    sampling_rate: int,
    padding: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if padding > 0:
        data = np.concatenate([data, np.array([0]*padding)])
        time_scale_rate = 1
    else:
        time_scale_rate = sampling_rate / \
            len(data) if sampling_rate > len(data) else 1

    cfc = _dft(data, sampling_rate)

    time_scale_frequency_index = [f*time_scale_rate for f in range(len(cfc))]

    return (cfc, np.array(time_scale_frequency_index))


def idft(
    data: np.ndarray,
    sampling_rate: int
) -> np.complex128:
    """culcurate dft

    Args:
        data (np.ndarray): single channel signal data.

    Return:
        complex
    """
    n_dft = len(data) * 2
    sigma = np.zeros(n_dft, dtype=np.complex128)
    for n, c in enumerate(data):
        omega = 2 * np.pi * n
        def e(n): return np.exp(1j*(omega*n/n_dft))
        sigma += np.array([c*e(n) for n in range(n_dft)])
        sigma += 0

    return sigma
