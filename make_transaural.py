import numpy as np
import soundfile as sf
import scipy.signal as sp
import argparse
import os
import util


def make_inverse_filter(H_ll, H_rl, H_lr, H_rr):
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

    h_ll = np.fft.irfft(H_ll)
    h_rl = np.fft.irfft(H_rl)
    h_lr = np.fft.irfft(H_lr)
    h_rr = np.fft.irfft(H_rr)

    tranceaural_l = sp.fftconvolve(sig_l, h_ll) + sp.fftconvolve(sig_r, h_rl)
    tranceaural_r = sp.fftconvolve(sig_l, h_lr) + sp.fftconvolve(sig_r, h_rr)

    tranceaural = np.array([tranceaural_l, tranceaural_r]).T

    return tranceaural


if __name__ == "__main__":
    """HRTFを用いてトランスオーラルを生成するプログラム
    使い方:
    python make_tranceaural.py {左右のHRTFを含むフォルダパス} {トランスオーラルにする元の音源}
    例 -> python make_tranceaural.py folderpath original.wav
    出力:
    - .wav
        - original_Hs.wav: イヤホンで聞いた時に、地下室の残響が再現されてるはずの音．音源とインパルス応答の畳込み
        - original_GHs.wav: sigに戻ってるはずの音．Hsとインパルス応答の逆行列のから求めたフィルタの畳み込み
        - original_Gs.wav: 地下室で聞いた時に、イヤホンで聞いたような音になるはずの音．音源とインパルス応答の逆行列のから求めたフィルタの畳み込み
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

    G_ll, G_rl, G_lr, G_rr = make_inverse_filter(H_ll, H_rl, H_lr, H_rr)
    Hs = make_transaural_sound(
        sig, H_ll, H_rl, H_lr, H_rr
    )  # イヤホンで聞いた時に、地下室の残響が再現されてるはず
    GHs = make_transaural_sound(Hs, G_ll, G_rl, G_lr, G_rr)  # sigに戻ってるはず
    Gs = make_transaural_sound(
        sig, G_ll, G_rl, G_lr, G_rr
    )  # 地下室で聞いた時に、イヤホンで聞いたような音になるはず
    print(G_ll * H_ll + G_lr * H_rl)
    print("\n")
    print(G_ll * H_rl + G_rl * H_rr)

    filename, _ = util.get_fileinfo(filepath_sound)
    sf.write(
        file=f"{folderpath}/{filename}_Hs.wav",
        data=Hs,
        samplerate=fs,
    )
    sf.write(
        file=f"{folderpath}/{filename}_GHs.wav",
        data=GHs,
        samplerate=fs,
    )
    sf.write(
        file=f"{folderpath}/{filename}_Gs.wav",
        data=Gs,
        samplerate=fs,
    )
