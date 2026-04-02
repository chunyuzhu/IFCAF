import torch
from thop import profile, clever_format
from models.yolo import CFTModel
from utils.general import check_yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = check_yaml("./models/INFAF/INFAF_yolov5l.yaml")
model = CFTModel(cfg).to(device).eval()

img1 = torch.randn(1, 3, 640, 640).to(device)
img2 = torch.randn(1, 3, 640, 640).to(device)

macs, params = profile(model, inputs=(img1,img2), verbose=False)  # 这里按我们 patch 的 list/tuple 输入
flops = 2 * macs

macs_s, params_s = clever_format([macs, params], "%.3f")
flops_s, = clever_format([flops], "%.3f")
print("Params:", params_s)
print("MACs  :", macs_s)
print("FLOPs :", flops_s, "(≈2*MACs)")
