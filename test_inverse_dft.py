from typing import List
from pathlib import Path

import pytest
import numpy as np
import matplotlib.pyplot as plt

from .dft import dft, idft


@pytest.mark.parametrize(
    """
    sampling_rate,
    signal_frequency,
    signal_amplitude,
    signal_phase,
    signal_length,
    padding,
    """,
    [
        (16, np.array([1]), np.array([1]), np.array([0]), 1, 0),
        (20, np.array([1]), np.array([1]), np.array([0]), 1, 0),
        (16, np.array([2]), np.array([1]), np.array([0]), 1, 0),
        (16, np.array([1]), np.array([2]), np.array([0]), 1, 0),

        (16, np.array([1]), np.array([1]), np.array([1]), 1, 0),
        (16, np.array([1]), np.array([1]), np.array([2]), 1, 0),
        (16, np.array([1]), np.array([1]), np.array([3]), 1, 0),

        (16, np.array([1]), np.array([1]), np.array([1]), 10, 0),
        (16, np.array([1]), np.array([1]), np.array([2]), 10, 0),
        (16, np.array([1]), np.array([1]), np.array([3]), 10, 0),

        (16, np.array([1]), np.array([1]), np.array([1]), 1, 16*9),
        (16, np.array([1]), np.array([1]), np.array([2]), 1, 16*9),
        (16, np.array([1]), np.array([1]), np.array([3]), 1, 16*9),

        (16, np.array([1]), np.array([1]), np.array([0]), 2, 0),
        (16, np.array([1]), np.array([1]), np.array([0]), 1, 16),
        (16, np.array([1]), np.array([1]), np.array([0]), 1, 10),


        (
            16,
            np.array([1, 3, 5]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
            1,
            0
        ),
        (
            800,
            np.array([1, 3, 5]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
            1,
            0
        ),
        (
            800,
            np.array([10, 30, 50]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
            1,
            0
        ),
        (
            800,
            np.array([1, 3, 5]),
            np.array([1, 0.5, 0.1]),
            np.array([0, 0, 0]),
            1,
            0
        ),
        (
            800,
            np.array([1, 3, 5]),
            np.array([1, 1, 1]),
            np.array([1, 1, 1]),
            1,
            0
        ),
        (
            800,
            np.array([1, 3, 5]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
            0.5,
            0
        ),
        (
            800,
            np.array([10, 100, 240]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
            0.5,
            0
        ),
        (
            800,
            np.array([10, 100, 240]),
            np.array([1, 1, 1]),
            np.array([1, 2, 3]),
            0.5,
            0
        ),
    ]
)
def test_dft(
    sampling_rate: int,
    signal_frequency: np.ndarray,
    signal_amplitude: np.ndarray,
    signal_phase: np.ndarray,
    signal_length: float,
    padding: int,
):
    """dft test

    Args:
        sampling_rate (int): _description_
        signal_frequency (np.ndarray): _description_
        signal_amplitude (np.ndarray): _description_
        signal_phase (float): _description_
        signal_length (float): _description_
        padding (int): _description_
    """
    # create signal
    signal = np.zeros(int(signal_length*sampling_rate))
    for f, a, p in zip(signal_frequency, signal_amplitude, signal_phase):
        signal += a * np.sin(
            2 * np.pi * f * np.arange(0, signal_length, 1/sampling_rate) + p)

    _dft, f_index = dft(signal, sampling_rate, padding)

    _idft = idft(_dft, sampling_rate)

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    ax[0].set_title("dft")
    ax[0].plot(f_index, np.abs(_dft))

    ax[1].set_title("idft")
    ax[1].plot(np.real(_idft))

    ax[2].set_title("source signal")
    ax[2].plot(signal)

    fig.subplots_adjust(left=0.08, right=0.98)
    script_dirpath = Path(__file__).parent
    plt.savefig(
        script_dirpath /
        f"test_log/idft/idft_sr{sampling_rate}_f{signal_frequency}_amp{signal_amplitude}_phs_{signal_phase}_pad{padding}_len{signal_length}.png"
    )
