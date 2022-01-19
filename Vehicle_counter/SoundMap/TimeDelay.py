import sys
import os

current = os.path.dirname(os.path.realpath("TimeDelay.ipynb"))
parent = os.path.dirname(current)
sys.path.append(parent)

from Vehicle_counter.DataPreparation.IDMT_open import import_dataset_of_records, import_record
from Vehicle_counter.SoundMap.gccestimating import GCC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def time_delay_3d(signal, sr=48000, lag=0.52, threshold=0.1):
    signal_left, signal_right = signal
    block_size = 0.2 * sr
    batch_size = 0.025 * sr
    size = int(sr * 0.2 * 2 - 1)
    time_delays = np.empty((0, size))
    for i in range(int((len(signal_left) - block_size) // batch_size + 1)):
        index_start = int(i * batch_size)
        index_end = int(index_start + block_size)
        gcc = GCC(signal_left[index_start:index_end], signal_right[index_start:index_end])
        time_delays = np.vstack((time_delays, gcc.cc()))
    time_delays = np.transpose(time_delays)
    centre = int(size // 2)
    lag_size = int(sr * lag // 1000)
    time_delays = time_delays[centre - lag_size:centre + lag_size]
    for i in range(len(time_delays)):
        for j in range(len(time_delays[0])):
            if time_delays[i][j] < threshold:
                time_delays[i][j] = 0
    return time_delays


def time_delay_2d(signal, sr=48000, lag=0.54, threshold=0.1):
    signal_left, signal_right = signal
    block_size = 0.2 * sr
    batch_size = 0.025 * sr
    size = int(sr * 0.2 * 2 - 1)
    time_delays = []
    for i in range(int((len(signal_left) - block_size) // batch_size + 1)):
        index_start = int(i * batch_size)
        index_end = int(index_start + block_size)
        gcc = GCC(signal_left[index_start:index_end], signal_right[index_start:index_end])
        delay = np.argmax(gcc.cc())
        time_delays.append(delay)
    centre = int(size // 2)
    for i in range(len(time_delays)):
        time_delays[i] = -(time_delays[i] - centre) * 1000 / sr
        if time_delays[i] < -lag or time_delays[i] > lag:
            time_delays[i] = 0
    return np.array(time_delays)


def plot_time_delays(time_delays):
    plt.imshow(time_delays, aspect='auto', norm=colors.Normalize())
    plt.show()
