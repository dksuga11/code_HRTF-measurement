import numpy as np
import soundfile as sf
import argparse
import os
import util


def conv_signal_and_hrir(sig, hrir):
    """信号と頭部インパルス応答を畳み込む関数
    Args:
        sig(ndarray): 畳み込む信号
        hrir(ndarray): 頭部インパルス応答
    Return:
        conv: 畳み込み結果
    """
    conv = np.convolve(sig, hrir)
    conv = conv / np.max(np.abs(conv))

    return conv


if __name__ == "__main__":
    """左右の耳元のインパルス応答を信号に畳み込んで，信号に残響を乗せるプログラム
    使い方:
        python conv_signal_ir.py {頭部インパルス応答を含んだフォルダのパス} {残響を乗せる音のファイルパス}
        例 -> python conv_signal_ir.py sp-l sound_source/piano.wav
        オプション引数として最後に｛--ext npy｝を追加することで，バイナリファイルで保存したhiriを使うことも可能
        例 -> python conv_signal_ir.py sp-l sound_source/piano.wav --ext npy
    注意:
        - 入力する頭部インパルス応答のファイル名は，calc_HRIR.pyで出力した名前を使用
        - 残響を乗せる音はモノラル限定
    """
    perser = argparse.ArgumentParser()
    perser.add_argument("folderpath", type=str)
    perser.add_argument("filepath_sound", type=str)
    perser.add_argument("--ext", default="wav", type=str)
    args = perser.parse_args()
    folderpath = args.folderpath
    filepath_sound = args.filepath_sound
    ext = args.ext

    # インパルス応答をwavファイルで入力する場合------
    if ext == "wav":
        ir_ear_l, _ = sf.read(f"{folderpath}/HRIR-l.wav")
        ir_ear_r, _ = sf.read(f"{folderpath}/HRIR-r.wav")
    # インパルス応答をバイナリファイルで入力する場合------
    if ext == "npy":
        ir_ear_l = np.load(f"{folderpath}/HRIR-l.npy")
        ir_ear_r = np.load(f"{folderpath}/HRIR-r.npy")

    in_sig, fs = sf.read(filepath_sound)

    out_sig_l = conv_signal_and_hrir(in_sig, ir_ear_l)
    out_sig_r = conv_signal_and_hrir(in_sig, ir_ear_r)
    collect_length = np.min([len(out_sig_l), len(out_sig_r)])
    out_sig = np.array([out_sig_l[:collect_length], out_sig_r[:collect_length]]).T

    filename, _ = util.get_fileinfo(filepath_sound)
    sf.write(
        file=f"{folderpath}/{filename}_convolve_hrir.wav", data=out_sig, samplerate=fs
    )
