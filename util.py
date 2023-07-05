import os
import numpy as np
from scipy import signal


def sync_time_difference(x, o, plot=False):  # estimate the discrete time difference τ
    """
    Parameters
    ----------
    x: array_like
        the recorded signal x[n]
    o: array_like
        the acoustic object signal o[n]
    plot: bool, optional (default=False)
        plot the cross-correlation between x[n] and o[n]

    Returns
    -------
    tau: int
        the discrete time difference
    """

    col = signal.fftconvolve(x, o[::-1])  # the cross-correlation between x[n] and o[n]
    tau = np.argmax(col) - (o.size - 1)  # [sample]

    if plot:
        import matplotlib.pyplot as plt

        t = np.arange(col.size - o.size)  # [sample]
        plt.plot(t, col[o.size :])
        plt.show()

    return tau


def get_fileinfo(filepath):
    """ファイル名や保存されているフォルダ名を返す
    Args:
        filepath(path):ファイルパス
    Return:
        filename(str):ファイル名
        folderpath(path):ファイルが保存されているフォルダ名
    """
    filename = os.path.basename(filepath)
    filename = os.path.splitext(filename)[0]
    folderpath = os.path.dirname(filepath)

    return filename, folderpath
