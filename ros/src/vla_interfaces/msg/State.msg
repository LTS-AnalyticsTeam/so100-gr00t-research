# ----- enum 定義 -----
uint8 NORMAL     = 0
uint8 ANOMALY    = 1
uint8 RECOVERING = 2

# ----- フィールド -----
uint8 state                          # NORMAL, ANOMALY, RECOVERING
vla_interfaces/Action action         # 行動
uint64 stamp                         # タイムスタンプ（sec + nsec）