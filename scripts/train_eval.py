"""TRAIN: fine-tuning de YOLO preentrenado, evaluación en test y latencia.

Uso (desde la raíz del repo, con data/processed ya generado):
    uv run python scripts/train_eval.py                  # entrena y evalúa
    uv run python scripts/train_eval.py --skip-train     # solo evalúa lo ya entrenado
Entrenar requiere GPU; con CPU tardaría días.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from vpc2.data.augmentation import load_policy

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RUNS = ROOT / "reports" / "runs"


def make_data_yaml() -> Path:
    """Crea un data.yaml con la ruta absoluta de ESTA máquina (evita rutas ajenas)."""
    cfg = yaml.safe_load((PROCESSED / "data.yaml").read_text())
    cfg.update(path=str(PROCESSED), train="images/train", val="images/val", test="images/test")
    out = PROCESSED / "data_local.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def measure_latency(model: YOLO, n: int = 200) -> dict:
    imgs = sorted((PROCESSED / "images" / "test").glob("*.*"))[:n]
    cuda = torch.cuda.is_available()
    for p in imgs[:20]:  # calentamiento
        model.predict(str(p), imgsz=640, verbose=False)
    tiempos = []
    for p in imgs:
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        model.predict(str(p), imgsz=640, verbose=False)
        if cuda:
            torch.cuda.synchronize()
        tiempos.append((time.perf_counter() - t) * 1000)
    ms = float(np.mean(tiempos))
    return {"ms_medio": round(ms, 2), "ms_mediana": round(float(np.median(tiempos)), 2),
            "FPS": round(1000 / ms, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--policy", default="proposed", help="arm de configs/augmentation.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    data = make_data_yaml()
    policy = load_policy(args.policy)
    metricas, latencias, por_clase = [], [], {}

    for w in args.weights:
        name = f"{Path(w).stem}_{args.policy}"
        if not args.skip_train:
            YOLO(w).train(data=str(data), epochs=args.epochs, imgsz=640, batch=args.batch,
                          seed=42, workers=2, project=str(RUNS), name=name,
                          exist_ok=True, **policy.train_kwargs())
        model = YOLO(str(RUNS / name / "weights" / "best.pt"))
        m = model.val(data=str(data), split="test", imgsz=640, plots=False)
        p, r = m.box.mp, m.box.mr
        metricas.append({"modelo": name, "mAP50": round(m.box.map50, 4),
                         "mAP50-95": round(m.box.map, 4), "precision": round(p, 4),
                         "recall": round(r, 4), "F1": round(2 * p * r / (p + r), 4)})
        por_clase[name] = dict(zip(m.names.values(), m.box.maps.round(4)))
        latencias.append({"modelo": name, **measure_latency(model)})

    RUNS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metricas).to_csv(RUNS / "tabla_metricas.csv", index=False)
    pd.DataFrame(latencias).to_csv(RUNS / "latencia.csv", index=False)
    pd.DataFrame(por_clase).to_csv(RUNS / "map_por_clase.csv")
    print(pd.DataFrame(metricas), pd.DataFrame(latencias), pd.DataFrame(por_clase), sep="\n\n")


if __name__ == "__main__":
    main()