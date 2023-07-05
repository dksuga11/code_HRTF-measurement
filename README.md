# code_HRTF-measurement
- `make_swept_sine.py`: 頭部インパルス応答を求めるときの測定用信号（swept-sine信号）を生成
- `calc_HRIR.py`: 測定した信号に逆信号を畳み込んで，以下を求める．
    - 左右の耳元のインパルス応答
    - 頭部中心にあたる位置のインパルス応答
    - 左右のHRIR
- `conv_signal_ir.py`: 左右の耳元のインパルス応答を信号を用いて信号に残響を乗せる
- `util.py`: よく使いそうな関数まとめ