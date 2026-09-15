"""Synthetic CPU integration check. Does NOT produce a deployable product model."""
import csv
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
import yaml
from PIL import Image
from engine.core import YAMLConfig
from engine.solver import TASKS
from src.infrastructure.ml.model_factory import ModelFactory


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="product_smoke_") as directory:
        root = Path(directory)
        config = yaml.safe_load(Path("configs/product_inspection.yml").read_text())
        config.update(device="cpu", epoches=1, early_stopping_patience=0,
                      output_dir=str(root / "output"), summary_dir=str(root / "summary"))
        # Random initialization is explicitly confined to this synthetic test.
        config[config["model"]]["backbone"]["weights_path"] = None
        for key in ("train_dataloader", "val_dataloader"):
            folder = root / key
            folder.mkdir()
            Image.new("RGB", (64, 64), "white").save(folder / "good.png")
            Image.new("RGB", (64, 64), "red").save(folder / "defect.png")
            manifest = folder / "annotations.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerows([["image", "labels"], ["good.png", "[]"], ["defect.png", '["DENT"]']])
            config[key]["total_batch_size"] = 2
            ds = config[key]["dataset"]
            ds.update(root=str(folder), annotations_path=str(manifest))
            ds["transforms"]["ops"][0]["size"] = [64, 64]
        path = root / "config.yml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        cfg = YAMLConfig(str(path))
        solver = TASKS[cfg.task](cfg)
        try:
            solver.fit()
            checkpoint = next((root / "summary").rglob("best.pth"))
            inspector = ModelFactory.build(str(path), str(checkpoint), "cpu")
            result = inspector.classify_image(str(root / "val_dataloader" / "good.png"))
            assert len(result.meta["scores"]) == 7
            assert result.coarse_label in {"PASS", "REVIEW", "FAIL"}
            # Exercise the production 384px transform/forward separately.
            config["val_dataloader"]["dataset"]["transforms"]["ops"][0]["size"] = [384, 384]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            production = ModelFactory.build(str(path), str(checkpoint), "cpu")
            assert len(production.classify_image(str(root / "val_dataloader" / "good.png")).meta["scores"]) == 7
            print("PASS: real DINOv3 CPU train/evaluate/checkpoint/reload/inference (64px and 384px); synthetic data only")
        finally:
            if solver.writer:
                solver.writer.close()


if __name__ == "__main__":
    main()
