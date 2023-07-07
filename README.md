# code_HRTF-measurement
- `make_swept_sine.py`: 頭部インパルス応答を求めるときの測定用信号（swept-sine信号）を生成
- `calc_HRIR.py`: 測定した信号に逆信号を畳み込んで，以下を求める．
    - 左右の耳元のインパルス応答
    - 頭部中心にあたる位置のインパルス応答
    - 左右のHRIR
- `conv_signal_ir.py`: 左右の耳元のインパルス応答と信号を畳み込んで残響が乗った信号のステレオ音を生成
- `conv_signal_hrir.py`: 左右の頭部インパルス応答と信号を畳み込みステレオ音を生成
- `util.py`: よく使いそうな関数まとめ