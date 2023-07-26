import numpy as np
import soundfile as sf
import scipy.signal as sp
import argparse
import os
import util


def make_transaural_filter(H_ll, H_lr, H_rl, H_rr):
    """トランスオーラルの生成に必要なフィルタを生成する
    Args:
        H_ll(ndarray): 左スピーカから左耳のまでのHRTF
        H_rl(ndarray): 右スピーカから左耳のまでのHRTF
        H_lr(ndarray): 左スピーカから右耳のまでのHRTF
        H_rr(ndarray): 右スピーカから右耳のまでのHRTF

    Return:
        G_ll(ndarray): 左耳に聞かせたい音(左目的音)を左スピーカーから出すときに最適化するフィルタ
        G_rl(ndarray): 右耳に聞かせたい音(右目的音)を左スピーカーから出すときに最適化するフィルタ
        G_lr(ndarray): 左目的音を右スピーカーから出すときに最適化するフィルタ
        G_rr(ndarray): 右目的音を右スピーカーから出すときに最適化するフィルタ
    """

    G_ll = H_rr / (H_ll * H_rr - H_rl * H_lr)
    G_rl = -H_rl / (H_ll * H_rr - H_rl * H_lr)
    G_lr = -H_lr / (H_ll * H_rr - H_rl * H_lr)
    G_rr = H_ll / (H_ll * H_rr - H_rl * H_lr)

    return G_ll, G_rl, G_lr, G_rr


def make_transaural_sound(sig, H_ll, H_rl, H_lr, H_rr):
    """トランスオーラルを生成
    Args:
        sig(ndarray):ステレオ信号
        H_ll(ndarray): 左スピーカから左耳のまでのHRTF
        H_rl(ndarray): 右スピーカから左耳のまでのHRTF
        H_lr(ndarray): 左スピーカから右耳のまでのHRTF
        H_rr(ndarray): 右スピーカから右耳のまでのHRTF
    return:
        transaural(ndarray):sigにフィルタを畳み込んだトランスオーラル
    """

    sig_l = sig.T[0]
    sig_r = sig.T[1]

    G_ll, G_rl, G_lr, G_rr = make_transaural_filter(H_ll, H_rl, H_lr, H_rr)
    g_ll = np.real(np.fft.irfft(G_ll))
    g_rl = np.real(np.fft.irfft(G_rl))
    g_lr = np.real(np.fft.irfft(G_lr))
    g_rr = np.real(np.fft.irfft(G_rr))

    tranceaural_l = sp.fftconvolve(sig_l, g_ll) + sp.fftconvolve(sig_r, g_rl)
    tranceaural_r = sp.fftconvolve(sig_l, g_lr) + sp.fftconvolve(sig_r, g_rr)

    tranceaural = np.array([tranceaural_l, tranceaural_r]).T

    return tranceaural


if __name__ == "__main__":
    """HRTFを用いてトランスオーラルを生成するプログラム
    使い方:
    python make_tranceaural.py {左右のHRTFを含むフォルダパス} {トランスオーラルにする元の音源}
    例 -> python make_tranceaural.py folderpath original.wav
    出力:
    - .wav
        - original.wavをトランスオーラルに変換した音源
    注意:
    - フォルダの中身（例のfoldrpath）には，左右のスピーカーごとに{sp-l}というフォルダを作成し，その中に左右のHRTFを{HRTF-l.npy}という名前で保存する
    """
    perser = argparse.ArgumentParser()
    perser.add_argument("folderpath", type=str)
    perser.add_argument("filepath_sound", type=str)
    args = perser.parse_args()

    folderpath = args.folderpath
    filepath_sound = args.filepath_sound
    H_ll = np.load(f"{folderpath}/sp-l/HRTF-l.npy")
    H_rl = np.load(f"{folderpath}/sp-r/HRTF-l.npy")
    H_lr = np.load(f"{folderpath}/sp-l/HRTF-r.npy")
    H_rr = np.load(f"{folderpath}/sp-r/HRTF-r.npy")
    sig, fs = sf.read(filepath_sound)

    tranceaural = make_transaural_sound(sig, H_ll, H_rl, H_lr, H_rr)

    filename, _ = util.get_fileinfo(filepath_sound)
    sf.write(
        file=f"{folderpath}/{filename}_tranceaural.wav", data=tranceaural, samplerate=fs
    )
