"""DGM4 contracts: labels, frozen features, metrics, and checkpoint round-trip."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from src.domain.dgm4_annotations import CLASS_NAMES, read_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for split in ("train", "val", "test"):
            for label in CLASS_NAMES:
                index = len(self.rows)
                path = self.root / split / label.lower() / "image.png"
                path.parent.mkdir(parents=True)
                image = Image.new("RGB", (32, 32), (index * 35, 20, 40))
                image.save(path)
                self.rows.append(dict(image=path.relative_to(self.root).as_posix(), split=split,
                    label=label, source_id=str(index), source_image=f"source/{index}.png",
                    pixel_sha256=hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
                    method="original" if label == "REAL" else "simswap",
                    fake_cls="orig" if label == "REAL" else "face_swap",
                    fake_image_box="[]" if label == "REAL" else "[1,2,10,12]"))
        self.write()

    def write(self):
        with (self.root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        (self.root / "report.json").write_text(json.dumps({"complete": True, "total": len(self.rows)}))

    def test_labels_and_provenance(self):
        self.assertEqual(CLASS_NAMES, ("REAL", "AI_EDITED"))
        self.assertEqual(len(read_manifest(self.root)), 6)

    def test_cross_split_source_rejected(self):
        self.rows[2]["source_id"] = self.rows[0]["source_id"]
        self.write()
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            read_manifest(self.root)

    def test_swapped_folder_label_rejected(self):
        self.rows[0]["label"] = "AI_EDITED"
        self.write()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            read_manifest(self.root)

    def test_text_only_manipulation_rejected(self):
        self.rows[1]["fake_cls"] = "text_swap"
        self.write()
        with self.assertRaisesRegex(ValueError, "face manipulation"):
            read_manifest(self.root)


class ModelTests(unittest.TestCase):
    def test_frozen_backbone_and_train_checkpoint_round_trip(self):
        import torch
        from engine.backbone.dgm4_classifier import DINOv3BinaryClassifier
        from engine.solver.dgm4_solver import train_head
        from src.infrastructure.ml.dgm4 import load_config, validate_checkpoint
        from src.shared.exceptions import ModelLoadError

        class TinyBackbone(torch.nn.Module):
            embed_dim = 3

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 3)
                self.dropout = torch.nn.Dropout(.5)

            def forward_features(self, x):
                value = self.projection(x.mean((2, 3)))
                return {"x_norm_clstoken": value, "x_norm_patchtokens": value[:, None, :]}

        torch.manual_seed(1)
        model = DINOv3BinaryClassifier(TinyBackbone())
        model.train()
        self.assertFalse(model.backbone.training)
        before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        x = model.extract_features(torch.rand(8, 3, 32, 32))
        y = torch.tensor([0, 1] * 4)
        self.assertEqual(tuple(model(torch.rand(2, 3, 32, 32)).shape), (2, 2))
        cfg = load_config()
        cfg.update(epochs=2, head_batch_size=4, early_stopping_patience=0)
        head_before = model.head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            train_head(model, (x, y), (x, y), cfg, Path(folder), {"smoke_only": True})
            state = torch.load(Path(folder) / "last.pth", weights_only=True)
            reloaded = DINOv3BinaryClassifier(TinyBackbone())
            reloaded.load_state_dict(state["model"], strict=True)
            self.assertTrue(torch.equal(model.head(x), reloaded.head(x)))
            self.assertFalse(torch.equal(head_before, model.head.weight))
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.backbone.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
            self.assertEqual(state["epoch"], 1)
            self.assertIn("optimizer", state)
            with self.assertRaisesRegex(ModelLoadError, "Synthetic"):
                validate_checkpoint(state, cfg)
            state["class_list"] = ["AI_EDITED", "REAL"]
            with self.assertRaisesRegex(ModelLoadError, "label order"):
                validate_checkpoint(state, cfg, allow_smoke=True)

    def test_metrics_and_softmax_decision(self):
        import torch
        from engine.solver.dgm4_solver import binary_metrics
        from src.infrastructure.ml.dgm4 import result_from_scores
        metrics = binary_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([.1, .8, .2, .9]))
        self.assertEqual(metrics["confusion_matrix_rows_true_cols_pred"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["false_positive_rate"], .5)
        self.assertEqual(metrics["false_negative_rate"], .5)
        self.assertAlmostEqual(metrics["roc_auc"], .75)
        self.assertEqual(result_from_scores([.5, .5])["label"], "AI_EDITED")
        self.assertEqual(result_from_scores([.8, .2])["label"], "REAL")
        with self.assertRaises(ValueError):
            result_from_scores([float("nan"), .2])


if __name__ == "__main__":
    unittest.main()
