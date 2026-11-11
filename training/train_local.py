"""CPU training of a first YOLO11n-seg model (the Colab notebook trains the full version on GPU).

python training/train_local.py --data dataset/data.yaml --epochs 12
Evaluates on the test split (simulator objects never seen in training), exports ONNX at
480 x 640 and registers the model in web/models/models.json and web/models/report.json.
"""
import argparse, json, os, shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["plastic", "glass", "cardboard", "metal", "paper", "organic", "other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolo11n-seg")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    run = ROOT / "runs" / a.model
    if a.resume and (run / "weights" / "last.pt").exists():
        YOLO(str(run / "weights" / "last.pt")).train(resume=True)
    else:
        YOLO(f"{a.model}.pt").train(
            data=a.data, imgsz=640, rect=False, epochs=a.epochs, batch=a.batch, patience=6, seed=0, workers=2,
            device="cpu", fraction=a.fraction, cos_lr=True, warmup_epochs=1,
            mosaic=0.3, close_mosaic=3, scale=0.15, translate=0.05, degrees=0.0, fliplr=0.5, flipud=0.5,
            hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, erasing=0.0, copy_paste=0.0,
            project=str(ROOT / "runs"), name=a.model, exist_ok=True, plots=True, amp=False)
    m = YOLO(str(run / "weights" / "best.pt"))
    r = m.val(data=a.data, split="test", imgsz=640, batch=8, device="cpu", project=str(ROOT / "runs"),
              name=f"{a.model}-test", exist_ok=True, plots=True)
    rep = {
        "trained_on": "cpu (first version)", "epochs": a.epochs,
        "box_map50": float(r.box.map50), "box_map50_95": float(r.box.map),
        "mask_map50": float(r.seg.map50), "mask_map50_95": float(r.seg.map),
        "mask_precision": float(r.seg.mp), "mask_recall": float(r.seg.mr),
        "per_class_mask_map50": {NAMES[int(c)]: float(v) for c, v in zip(r.ap_class_index, r.seg.ap50)},
        "confusion_matrix_rows_pred_cols_true": r.confusion_matrix.matrix.tolist(),
        "params": int(sum(p.numel() for p in m.model.parameters())),
    }
    onnx_path = m.export(format="onnx", imgsz=(480, 640), opset=12, simplify=True, nms=False, dynamic=False, half=False)
    dst = ROOT / "web" / "models" / f"wastevision-{a.model}.onnx"
    shutil.copy(onnx_path, dst)
    rep["onnx_mb"] = os.path.getsize(dst) / 1e6
    mj = ROOT / "web" / "models" / "models.json"
    models = json.loads(mj.read_text())["models"]
    any_default = False
    for e in models:
        if e["id"] == a.model:
            e["available"] = True
        any_default |= bool(e.get("available") and e.get("default"))
    if not any_default:
        for e in models:
            e["default"] = e["id"] == a.model
    mj.write_text(json.dumps({"models": models}, indent=2))
    rj = ROOT / "web" / "models" / "report.json"
    report = json.loads(rj.read_text()) if rj.exists() else {}
    report[a.model] = rep
    rj.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "confusion_matrix_rows_pred_cols_true"}, indent=1))


if __name__ == "__main__":
    main()
