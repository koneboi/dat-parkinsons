"""Final gate for submission_v10.zip — everything must pass before submission."""
import zipfile, json, io, torch, hashlib
from src.train_resnet import build_model

ZIP = "submission_v10.zip"
z = zipfile.ZipFile(ZIP)
names = set(z.namelist())
print(f"[1] zip entries: {len(names)}")
assert "main.py" in names and {"src/__init__.py","src/align.py","src/model.py","src/preprocess.py"} <= names
print("[2] code files present: OK")

c = json.loads(z.read("model/calib.json"))
assert len(c["members"]) == 13 and len(c["weights"]) == 13
assert len(c["zmean"]) == 13 and len(c["zstd"]) == 13
assert 0 < c["a"] < 50 and abs(c["b"]) < 20
assert c["blend_logloss"] < 0.30
print(f"[3] calib valid: a={c['a']:.4f} b={c['b']:.4f} LL={c['blend_logloss']:.4f} AUC={c['blend_auc']:.4f}")

nf = 0
for m in c["members"]:
    model = build_model(backbone=m["backbone"], in_channels=6 if m.get("stack")=="aip" else 3)
    mk = set(model.state_dict().keys())
    for k in range(5):
        fn = f"model/{m['name']}_fold_{k}.pt"
        st = torch.load(io.BytesIO(z.read(fn)), map_location="cpu", weights_only=True)
        assert set(st.keys()) == mk, f"keys mismatch {fn}"
        nf += 1
assert nf == 65
print(f"[4] all {nf} fold weights load & match architecture: OK")

hs = lambda f: hashlib.sha256(f).hexdigest()
print(f"[5] file SHA256: {hs(open(ZIP,'rb').read())}")
print("\n===== FINAL GATE PASSED — submission_v10.zip is ready =====")