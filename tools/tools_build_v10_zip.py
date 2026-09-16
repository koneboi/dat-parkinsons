"""Build submission_v10.zip: v9 12 members + weights_effb0_s1234, calib_v10.json."""
import json, zipfile
from pathlib import Path

MEMBERS = ["weights_r50_s7","weights_dense_s999","weights_str120_dense","weights_r50_s123",
           "weights_r50_str160_128","weights_r50_mipaip","weights_r50_ema","weights_dense_ema",
           "weights_str120_r50","weights_r50_s999","weights_dense_str160_128","weights_r18_s999",
           "weights_effb0_s1234"]

SRC = Path("submission_src")
calib = json.load(open("weights_final_v6_testcalib/calib_v10.json"))
out = "submission_v10.zip"

def member_entry(name, c):
    m = next((x for x in c["members"] if x["name"] == name), None)
    if isinstance(m, dict):
        return {"name": name, "backbone": m.get("backbone","resnet18"),
                "size": int(m.get("size",128)), "preproc": m.get("preproc","brain200"),
                "proj": m.get("proj","mip"), "stack": m.get("stack"),
                "tta": bool(m.get("tta", True))}
    return {"name": name, "backbone":"resnet18", "size":128,
            "preproc":"brain200", "proj":"mip", "stack":None, "tta":True}

newcal = {k: calib[k] for k in ["a","b","zmean","zstd","weights"]}
newcal["members"] = [member_entry(n, calib) for n in MEMBERS]
for k in ["blend_auc","blend_logloss","fix_note"]:
    if k in calib:
        newcal[k] = calib[k]

with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
    zf.writestr("main.py", (SRC/"main.py").read_bytes())
    for py in sorted((SRC/"src").glob("*.py")):
        zf.writestr(f"src/{py.name}", py.read_bytes())
    n_fold = 0
    for name in MEMBERS:
        mdir = Path(name)
        if not mdir.exists():
            print(f"!! missing dir {name}")
            continue
        for pt in sorted(mdir.glob("fold_*.pt")):
            zf.writestr(f"model/{name}_{pt.name}", pt.read_bytes())
            n_fold += 1
    zf.writestr("model/calib.json", json.dumps(newcal, indent=2))

sz = Path(out).stat().st_size / (1024**3)
print(f"Wrote {out} ({sz:.2f} GB) with {n_fold} fold weights")
print("calib members:", [m["name"]+f":tta={m['tta']}" for m in newcal["members"]])