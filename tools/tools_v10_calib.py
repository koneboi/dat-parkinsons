"""Build v10 calib.json: v9's 12 members + weights_effb0_s1234 (diverse arch).
Same honest single-fold OOF on the exact runtime-TTA scale (flips [3],[2])."""
import json
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from src.train_resnet import fit_sigmoid_calibration

d12 = np.load("oof_v9_runtime.npz")
dnew = np.load("oof_effb0.npz")
y = d12["y"]
assert np.allclose(y, dnew["y"])

base = ["weights_r50_s7", "weights_dense_s999", "weights_str120_dense", "weights_r50_s123",
        "weights_r50_str160_128", "weights_r50_mipaip", "weights_r50_ema", "weights_dense_ema",
        "weights_str120_r50", "weights_r50_s999", "weights_dense_str160_128", "weights_r18_s999"]
extra = ["weights_effb0_s1234"]
members = base + extra

def load(m):
    k = m + "_tta" if (m + "_tta") in dnew.files else m + "_tta"
    if (m + "_tta") in dnew.files:
        return dnew[m + "_tta"], dnew[m + "_notta"]
    return d12[m + "_tta"], d12[m + "_notta"]

oofs, use_tta = [], {}
for mm in members:
    o1, o2 = load(mm)
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
print(f"\nv10 blend: AUC {auc:.4f} LL {ll:.4f}  (a={a:.4f} b={b:.4f})")

newcal = json.load(open("weights_final_v6_testcalib/calib_v9.json"))
newcal["zmean"] = means
newcal["zstd"] = stds
newcal["a"] = a
newcal["b"] = b
newcal["blend_auc"] = auc
newcal["blend_logloss"] = ll
newcal["weights"] = [1.0] * len(members)
newcal["fix_note"] = ("v10: v9 (correct runtime flips [3],[2]) + weights_effb0_s1234 "
                      "(efficientnet_b0, diverse arch)")
mm = {m["name"]: m for m in newcal["members"]}
for name in members:
    if name in mm:
        mm[name]["tta"] = bool(use_tta[name])
    else:
        newcal["members"].append({"name": name, "seed": 1234, "size": 128,
                                  "backbone": "efficientnet_b0", "proj": "mip",
                                  "stack": None, "preproc": "brain200",
                                  "tta": bool(use_tta[name])})

json.dump(newcal, open("weights_final_v6_testcalib/calib_v10.json", "w"), indent=2)
print("saved -> weights_final_v6_testcalib/calib_v10.json")