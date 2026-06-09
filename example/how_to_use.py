from tkinter.scrolledtext import example

from court_corner.pipeline import CourtCornerPipeline

pipe = CourtCornerPipeline("../weight/best.pt", yolo_conf=0.4, corner_conf=0.6)
result = pipe.run("../datasets/real/labs/frame_00001.png")
for cid, x, y, conf in result.corners_as_tuples():
    print(cid, x, y, conf)


### print results
#
# [Stage1] YOLO 模型 class 名稱: {0: 'L', 1: 'T', 2: 'X'}
# [Stage1] 偵測到 12 個交點  (L=1, T=5, X=6)  conf≥0.4
# [Stage2/line] H 求解成功  method=link-enum+steger-refine+hungarian-refit  lc=1.000  tc=1.000  線支持=0.83  steger_refined=10  交點=12
# [Stage3] 由 12 個交點生成 34 個角點候選
# [Stage4] 完成：輸出 34 個角點 （strong 32 + weak 2，hidden 0；候選 34，門檻 conf≥0.6） ｜處理時間 2.46s
# 77 619.5052490234375 271.8609619140625 0.9500775776699159
# 43 224.48263549804688 330.4573974609375 0.90388470712111
# 79 622.7176513671875 273.92657470703125 0.9037803487928472
# ...
# ###


