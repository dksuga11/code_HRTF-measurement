import numpy as np
import scipy.signal as sp
import soundfile as sf
import argparse
import pathlib
import matplotlib.pyplot as plt


def swept_sine(flag):
    """swept-sine信号と逆swept-sine信号を生成
    Args:
        flag(int):1ならswept-sine, -1なら逆swept-sine
    Return
        sewpt:swept-sine信号か逆swept-sine信号
    """
    n = 15
    N = 2**n
    # scale = 10000

    Swept = np.zeros([N], dtype="complex_")
    for k in range(0, N // 2):
        kk = k + 1
        Swept[kk] = (
            np.cos(np.pi * k * k / N + 0.5 * np.pi * k)
            - np.sin(np.pi * k * k / N + 0.5 * np.pi * k) * 1j * flag
        )

    for k in range(N // 2, N - 1):
        kk = k + 1
        Swept[kk] = np.conj(Swept[N - kk])

    swept = np.fft.ifft(Swept)
    swept = swept / np.max(np.abs(np.real(swept)))
    swept = np.real(swept)

    # plt.figure(figsize=[6.0, 4.0])
    # plt.plot(np.abs(Swept))
    # plt.savefig("sweptsine_amp")
    # plt.figure(figsize=[6.0, 4.0])
    # plt.plot(np.angle(Swept))
    # plt.savefig("sweptsine_phase")

    return swept


if __name__ == "__main__":
    """頭部インパルス応答を求めるときの測定用信号（swept-sine信号）を生成するプログラム
    使い方:
        python make_swept_sine.py {flag(1 or -1)}
        例 -> python make_swept_sine.py 1
    注意:
        - flagの値が1ならswept-sine信号，-1なら逆swept-sine信号を生成
    """
    perser = argparse.ArgumentParser()
    perser.add_argument("flag", choices=["1", "-1"])
    args = perser.parse_args()
    flag = int(args.flag)

    swept = swept_sine(flag)
    if flag == 1:
        filename = "swept-sine"
    else:
        filename = "inverse_swept-sine"
    plt.figure(figsize=[6.0, 4.0])
    plt.plot(swept)
    plt.title(filename)
    plt.savefig(f"../swept-sine/{filename}_waveform.png")

    sf.write(file=f"../swept-sine/{filename}.wav", data=swept, samplerate=48000)
