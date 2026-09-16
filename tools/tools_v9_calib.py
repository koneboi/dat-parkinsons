"""Build v9 calib.json on the CORRECT runtime TTA scale (flips dims [3],[2]).
The v7/v8 calib was fit on OOF computed with channel-flip TTA (dims [2],[1]),
which main.py never uses at runtime -> z-stats/Platt mismatched -> 0.3427.
This refits everything on honest single-fold OOF with the exact runtime flips.
"""
import json
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from src.train_resnet import fit_sigmoid_calibration

d = np.load("/tmp/opencode/v9_oof.npz")
y = d["y"]
members = ["weights_r50_s7", "weights_dense_s999", "weights_str120_dense", "weights_r50_s123",
           "weights_r50_str160_128", "weights_r50_mipaip", "weights_r50_ema", "weights_dense_ema",
           "weights_str120_r50", "weights_r50_s999", "weights_dense_str160_128", "weights_r18_s999"]

oofs, use_tta = [], {}
for mm in members:
    k = mm.replace("/", "_")
    o1, o2 = d[f"{k}_tta"], d[f"{k}_notta"]
    a1, a2 = roc_auc_score(y, o1), roc_auc_score(y, o2)
    use_tta[mm] = a1 >= a2
    o = o1 if use_tta[mm] else o2
    oofs.append(o)
    print(f"{mm:26s} tta AUC {a1:.4f} | no-tta {a2:.4f} -> {'TTA' if use_tta[mm] else 'no-TTA'}   std {o.std():.3f}")

means = [float(o.mean()) for o in oofs]
stds = [float(o.std()) for o in oofs]

blend = np.zeros(len(y), dtype=np.float64)
for o, m, s in zip(oofs, means, stds):
    blend += (o - m) / (s + 1e-9)
blend /= len(oofs)

a, b = fit_sigmoid_calibration(blend, y)
p = np.clip(1 / (1 + np.exp(-(a * blend + b))), 1e-7, 1 - 1e-7)
ll = log_loss(y, p)
auc = roc_auc_score(y, blend)
print(f"\nv9 blend (runtime-TTA scale): AUC {auc:.4f} LL {ll:.4f}")
print(f"  a={a:.4f} b={b:.4f}")

newcal = json.load(open("weights_final_v6_testcalib/calib.json"))
newcal["zmean"] = means
newcal["zstd"] = stds
newcal["a"] = a
newcal["b"] = b
newcal["blend_auc"] = auc
newcal["blend_logloss"] = ll
newcal["fix_note"] = ("v9: OOF recomputed with EXACT main.py member_logits TTA "
                      "(flip dims [3],[2]) - v7/v8 used channel-flip [2],[1] which "
                      "runtime never applies, corrupting z-stats and Platt.")
for m in newcal["members"]:
    name = m["name"]
    m["tta"] = bool(use_tta[name])

json.dump(newcal, open("weights_final_v6_testcalib/calib_v9.json", "w"), indent=2)
print("saved -> weights_final_v6_testcalib/calib_v9.json")