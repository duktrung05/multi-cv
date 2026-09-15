import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image
from src.domain.inspection import inspection_result, aggregate_frames
from src.domain.inspection_annotations import read_annotations


class InspectionPolicyTests(unittest.TestCase):
    def test_dino_key_bias_mask_is_finite_and_masks_only_keys(self):
        import torch
        from engine.backbone.dinov3.layers.attention import LinearKMaskedBias
        layer = LinearKMaskedBias(4, 12)
        self.assertEqual(layer.bias_mask.tolist(), [1.] * 4 + [0.] * 4 + [1.] * 4)
        self.assertTrue(torch.isfinite(layer(torch.ones(1, 4))).all())

    def test_threshold_boundaries_and_multilabel(self):
        for score, decision in [(0, "PASS"), (.299, "PASS"), (.3, "REVIEW"), (.699, "REVIEW"), (.7, "FAIL"), (1, "FAIL")]:
            self.assertEqual(inspection_result({"DENT": score}).coarse_label, decision)
        result = inspection_result({"DENT": .9, "CRACK": .8, "DIRT": .4})
        self.assertEqual(result.meta["defects"], ["DENT", "CRACK"])
        self.assertEqual(result.meta["suspected_defects"], ["DIRT"])

    def test_invalid_scores_and_thresholds(self):
        for scores in [{}, {"DENT": float("nan")}, {"DENT": float("inf")}, {"DENT": -1}]:
            with self.assertRaises(ValueError):
                inspection_result(scores)
        with self.assertRaises(ValueError):
            inspection_result({"DENT": .2}, .7, .3)

    def test_video_preserves_defects_from_different_frames(self):
        results = [inspection_result({"DENT": .95, "CRACK": .1}),
                   inspection_result({"DENT": .01, "CRACK": .9}),
                   inspection_result({"DENT": .01, "CRACK": .01})]
        result = aggregate_frames(results, ["a", "b", "c"])
        self.assertEqual(result.coarse_label, "FAIL")
        self.assertEqual(result.meta["defects"], ["DENT", "CRACK"])
        self.assertEqual(result.meta["worst_frame"], "a")


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (16, 16)).save(self.root / "good.png")
        Image.new("RGB", (16, 16), "red").save(self.root / "bad.png")
        self.csv = self.root / "labels.csv"

    def write_rows(self, rows):
        with self.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "labels"])
            writer.writerows(rows)

    def test_normal_and_multiple_defects(self):
        self.write_rows([["good.png", "[]"], ["bad.png", '["CRACK","DENT"]']])
        samples = read_annotations(self.root, self.csv, ["DENT", "CRACK"])
        self.assertEqual([s[1] for s in samples], [[0., 0.], [1., 1.]])

    def test_reject_missing_unknown_and_duplicate(self):
        for rows in [[["good.png", ""]], [["good.png", '["UNKNOWN"]']],
                     [["absent.png", "[]"]], [["good.png", "[]"], ["good.png", "[]"]],
                     [["../outside.png", "[]"]]]:
            self.write_rows(rows)
            with self.assertRaises(ValueError):
                read_annotations(self.root, self.csv, ["DENT"])

    def test_dataset_transform_and_learning_step(self):
        import torch
        from torchvision.transforms import Compose, ToTensor, Resize
        from engine.data.dataset.product_inspection import ProductInspectionDataset
        from engine.backbone.multilabel_classification_adapter import DINOv3STAsMultiLabelClassifier
        from engine.solver.multilabel_criterion import MultiLabelBCELoss
        self.write_rows([["good.png", "[]"], ["bad.png", '["DENT"]']])
        ds = ProductInspectionDataset(self.root, self.csv, ["DENT", "CRACK"], Compose([Resize((32, 32)), ToTensor()]))

        class SmallBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = torch.nn.ModuleList([torch.nn.Conv2d(3, 8, 1)])

            def forward(self, x):
                y = self.convs[0](x)
                return y, y, y

        model = DINOv3STAsMultiLabelClassifier(SmallBackbone(), 2, pooling="gap")
        x = torch.stack([ds[i][0] for i in range(2)])
        y = torch.stack(ds.targets)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        before = model.head[-1].weight.detach().clone()
        loss = MultiLabelBCELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.head[-1].weight))
        self.assertEqual(tuple(model(x).shape), (2, 2))


class ApiTests(unittest.TestCase):
    def test_image_api_returns_all_defects_and_rejects_invalid_image(self):
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.application.services.classification_service import ClassificationService
        with tempfile.TemporaryDirectory() as directory:
            result = inspection_result({"DENT": .9, "CRACK": .8})
            classifier = SimpleNamespace(classify_image=lambda *a, **k: result)
            container = SimpleNamespace(settings=SimpleNamespace(tmp_dir=directory),
                workers=SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
                classifier_service=ClassificationService(classifier))
            image = io.BytesIO()
            Image.new("RGB", (16, 16)).save(image, format="PNG")
            with TestClient(create_app(lambda settings: container)) as client:
                response = client.post("/inspect/image", files={"file": ("product.png", image.getvalue(), "image/png")})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["ai_result"]["defects"], ["DENT", "CRACK"])
                invalid = client.post("/inspect/image", files={"file": ("bad.png", b"not image", "image/png")})
                self.assertEqual(invalid.status_code, 422)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_checkpoint_does_not_start_random_model(self):
        from src.infrastructure.ml.model_factory import ModelFactory
        from src.shared.exceptions import ModelLoadError
        with self.assertRaises(ModelLoadError):
            ModelFactory.build("configs/product_inspection.yml", "does-not-exist.pth")

    def test_checkpoint_label_order_is_verified(self):
        import torch
        from src.domain.inspection import DEFECT_LABELS
        from src.infrastructure.ml.model_factory import ModelFactory
        from src.shared.exceptions import ModelLoadError
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "wrong.pth"
            torch.save({"model": {}, "task_domain": "product_inspection",
                        "class_list": list(reversed(DEFECT_LABELS))}, checkpoint)
            with self.assertRaisesRegex(ModelLoadError, "label order"):
                ModelFactory.build("configs/product_inspection.yml", str(checkpoint))

    def test_video_api_aggregates_and_cleans_temporary_files(self):
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.application.services.classification_service import ClassificationService
        with tempfile.TemporaryDirectory() as directory:
            def extract(video, output, frame_step):
                return ["frame1", "frame2"]

            def classify(path, **kwargs):
                return inspection_result({"DENT": .9 if path == "frame1" else .01,
                                          "CRACK": .8 if path == "frame2" else .01})

            container = SimpleNamespace(settings=SimpleNamespace(tmp_dir=directory, frame_step=24),
                workers=SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
                storage=SimpleNamespace(download=lambda *args: None),
                video=SimpleNamespace(extract_keyframes=extract),
                classifier_service=ClassificationService(SimpleNamespace(classify_image=classify)))
            with TestClient(create_app(lambda settings: container)) as client:
                response = client.post("/inspect/video", json={"url": "https://example.com/test.mp4"})
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()["ai_result"]
                self.assertEqual(payload["defects"], ["DENT", "CRACK"])
                self.assertEqual(payload["worst_frame_index"], 0)
                self.assertNotIn("worst_frame", payload)
                invalid = client.post("/inspect/video", json={"url": "https://example.com/test.mp4", "frame_step": 0})
                self.assertEqual(invalid.status_code, 422)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
