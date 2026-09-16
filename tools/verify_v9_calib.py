import json
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

calib = json.load(open('weights_final_v6_testcalib/calib_v9.json'))
d = np.load('/tmp/opencode/v9_oof.npz')
y = d['y']

members = []
for m in calib['members']:
    name = m['name']
    key = name.replace('/', '_') + ('_tta' if m.get('tta', True) else '_notta')
    members.append((name, key, d[key]))

# Check z-mean/z-std against actual OOF stats
print("z-stat check (calib vs actual OOF):")
ok = True
for i, (name, key, o) in enumerate(members):
    zm, zs = float(calib['zmean'][i]), float(calib['zstd'][i])
    om, os_ = float(o.mean()), float(o.std())
    match = abs(zm - om) < 1e-4 and abs(zs - os_) < 1e-3
    if not match:
        ok = False
        print(f"  {name}: calib zm={zm:.4f} zs={zs:.4f} | oof mean={om:.4f} std={os_:.4f}  MISMATCH")
print("  all z-stats match OOF stats" if ok else "  !! z-stat mismatches found")

# Rebuild blend exactly as runtime: z-score each member, mean over members (weights=1)
w = np.array(calib['weights'], dtype=np.float64)
wsum = w.sum() + 1e-9
blend = np.zeros(len(y), dtype=np.float64)
for i, (name, key, o) in enumerate(members):
    blend += w[i] * (o.astype(np.float64) - float(calib['zmean'][i])) / (float(calib['zstd'][i]) + 1e-9)
mean_logit = blend / wsum
p = np.clip(1.0 / (1.0 + np.exp(-(calib['a'] * mean_logit + calib['b']))), 1e-7, 1 - 1e-7)
print(f"\nweights: {w}")
print(f"recomputed OOF LL: {log_loss(y, p):.4f}  AUC {roc_auc_score(y, mean_logit):.4f}")
print(f"stored calib:      LL {calib['blend_logloss']:.4f}  AUC {calib['blend_auc']:.4f}  a={calib['a']:.4f} b={calib['b']:.4f}")