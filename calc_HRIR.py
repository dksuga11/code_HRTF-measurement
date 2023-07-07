import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import argparse
import os
import make_swept_sine as mss
import util


def calc_IR(y):
    """耳元で収録したswept-sibe信号を逆swept-sine信号と畳み込み，インパルス応答を求めるプログラム
    Args:
        y(ndarray):収録信号
    Return:
        hrir(ndarray):インパルス応答
    """
    inv_swept = mss.swept_sine(flag=-1)
    swept = mss.swept_sine(flag=1)
    t_sync = util.sync_time_difference(y, swept)
    ir = np.convolve(y[t_sync:], inv_swept)
    ir = np.convolve(y, inv_swept)
    return ir

def find_IR(y_l, y_r, y_center):
    ir_ear_l = calc_IR(y_l)
    ir_ear_r = calc_IR(y_r)
    ir_center = calc_IR(y_center)

    ir_l_max = np.max(np.abs(ir_ear_l))
    ir_r_max = np.max(np.abs(ir_ear_r))
    ir_center_max = np.max(np.abs(ir_center))
    max = np.max([ir_l_max, ir_r_max, ir_center_max])

    ir_ear_l = ir_ear_l / max
    ir_ear_r = ir_ear_r / max
    ir_center = ir_center / max

    return ir_ear_l, ir_ear_r, ir_center


def calc_HRIR(ir_ear, ir_center, fs):
    """HRIRを求める関数
    Args:
        ir_ear(ndarray): 耳元のインパルス応答
        ir_center(ndarray): 頭部の中心に当たる位置のインパルス応答
        fs(int): サンプリング周波数（インパルス応答の時間同期に使用）
    Return:
        hrir: 頭部インパルス応答
    """
    t_sync_ear_bigin = int(np.argmax(np.abs(ir_ear)) - 50)
    t_sync_ear_end = int(np.argmax(np.abs(ir_ear)) + 128)
    t_sync_center_bigin = int(np.argmax(np.abs(ir_center)) - 50)
    t_sync_center_end = int(np.argmax(np.abs(ir_center)) + 128)
    temp = 512-(50+128)
    zeros = np.zeros(temp)
    ir_ear = np.append(ir_ear[t_sync_ear_bigin:t_sync_ear_end], zeros)
    ir_center = np.append(ir_center[t_sync_center_bigin:t_sync_center_end], zeros)

    Hrtf = np.fft.fft(ir_ear) / np.fft.fft(ir_center)
    hrir = np.real(np.fft.ifft(Hrtf))
    hrir = hrir / np.max(np.abs(hrir))

    return hrir


def plot_impuls(y, folderpath, filename):
    os.makedirs(f"{folderpath}/image", exist_ok=True)
    plt.figure(figsize=[6.0, 4.0])
    plt.plot(y)
    plt.ylim([-1*np.max(np.abs(y)), np.max(np.abs(y))])
    # plt.ylim([-1.2 ,1.2])
    plt.title(f"implus_responce_{filename}")
    plt.savefig(f"{folderpath}/image/{filename}.png")


if __name__ == "__main__":
    """測定した信号に逆信号を畳み込んで，HRIRを求めるプログラム
    使い方:
    python calc_HRIR.py {左右の測定信号（モノラル）を含んだフォルダのパス}
    例 -> python calc_HRIR.py sp-l.py
    出力:
    - .wav
        - 左右の耳元でのインパルス応答 (IR_ear-l, IR_ear-r)
        - 頭部中心にあたる位置でのインパルス応答 (IR_center)
        - 左右のHRIR (HRIR_l, HRIR_rs)
    - .npy
        - .wavと同じ
    - .png
        - .wavの波形(5つ)
    注意:
    - フォルダに入れる左右の測定信号のファイル名は，左耳での測定信号なら「recorded_ear-l.wav」，
      右耳での測定信号なら「recorded_ear-r.wav」，
      頭部の中心にあたる位置での測定信号なら「recorded_center.wav」とする
    """

    perser = argparse.ArgumentParser()
    perser.add_argument("folderpath", type=str)
    args = perser.parse_args()

    folderpath = args.folderpath
    y_l, fs = sf.read(f"{folderpath}/recorded_ear-l.wav")
    y_r, _ = sf.read(f"{folderpath}/recorded_ear-r.wav")
    y_center, _ = sf.read(f"{folderpath}/recorded_center.wav")

    # 耳元と頭部中心あたる位置のインパルス応答の計算と出力
    ir_ear_l, ir_ear_r, ir_center = find_IR(y_l, y_r, y_center)

    plot_impuls(ir_ear_l, folderpath, filename="ir_ear-l")
    plot_impuls(ir_ear_r, folderpath, filename="ir_ear-r")
    plot_impuls(ir_center, folderpath, filename="ir_center")

    np.save(f"{folderpath}/IR_ear-l", ir_ear_l)
    np.save(f"{folderpath}/IR_ear-r", ir_ear_r)
    np.save(f"{folderpath}/IR_center", ir_center)
    sf.write(file=f"{folderpath}/IR_ear-l.wav", data=ir_ear_l, samplerate=fs)
    sf.write(file=f"{folderpath}/IR_ear-r.wav", data=ir_ear_r, samplerate=fs)
    sf.write(file=f"{folderpath}/IR_center.wav", data=ir_center, samplerate=fs)

    # 頭部インパルス応答の計算と出力
    hrir_l = calc_HRIR(ir_ear_l, ir_center, fs)
    hrir_r = calc_HRIR(ir_ear_r, ir_center, fs)

    plot_impuls(hrir_l, folderpath, filename="hrir_l")
    plot_impuls(hrir_r, folderpath, filename="hrir_r")

    np.save(f"{folderpath}/HRIR-l", hrir_l)
    np.save(f"{folderpath}/HRIR-r", hrir_r)
    sf.write(file=f"{folderpath}/HRIR-l.wav", data=hrir_l, samplerate=fs)
    sf.write(file=f"{folderpath}/HRIR-r.wav", data=hrir_r, samplerate=fs)
