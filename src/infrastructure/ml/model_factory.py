"""Load a trained product checkpoint. No moderation model is used."""
from __future__ import annotations
import copy
from pathlib import Path
from src.domain.inspection import inspection_result
from src.shared.exceptions import ModelLoadError


class ProductInspector:
    def __init__(self, model, class_names, transform, device, review_threshold, fail_threshold):
        self.model, self.class_names = model, class_names
        self.transform, self.device = transform, device
        self.review_threshold, self.fail_threshold = review_threshold, fail_threshold

    def classify_image(self, image_path: str, criteria: int = 0):
        import torch
        from PIL import Image, ImageOps
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            if logits.shape != (1, len(self.class_names)):
                raise ValueError("Checkpoint output does not match defect labels")
            probabilities = logits.float().sigmoid()[0].cpu().tolist()
        return inspection_result(dict(zip(self.class_names, probabilities)),
                                 self.review_threshold, self.fail_threshold, criteria)


class ModelFactory:
    @staticmethod
    def build(config: str, checkpoint: str, device: str = "auto") -> ProductInspector:
        for path, name in ((config, "config"), (checkpoint, "trained product checkpoint")):
            if not Path(path).is_file():
                raise ModelLoadError(f"Missing {name}: {path}. See README.md for product training setup.")
        import torch
        from engine.core import YAMLConfig
        from engine.data.transforms.container import Compose
        cfg = YAMLConfig(config)
        y = cfg.yaml_cfg
        if y.get("model") != "DINOv3STAsMultiLabelClassifier":
            raise ModelLoadError("Product inspection requires DINOv3STAsMultiLabelClassifier")
        names = list(y.get("class_list") or [])
        if not names or len(set(names)) != len(names) or len(names) != y.get("num_classes"):
            raise ModelLoadError("Config must define unique class_list matching num_classes")
        review = float(y.get("inspection_review_threshold", 0.3))
        fail = float(y.get("inspection_fail_threshold", 0.7))
        inspection_result(dict.fromkeys(names, 0.0), review, fail)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("class_list") != names or state.get("task_domain") != "product_inspection":
            raise ModelLoadError("Checkpoint must be trained for product_inspection with the exact configured label order")
        y[y["model"]]["backbone"]["weights_path"] = None
        model = cfg.model
        model.load_state_dict(state["model"], strict=True)
        resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != "auto":
            resolved = device
        model.to(resolved).eval()
        spec = copy.deepcopy(y["val_dataloader"]["dataset"]["transforms"])
        transform = Compose(ops=spec["ops"], policy=spec.get("policy"))
        return ProductInspector(model, names, transform, resolved, review, fail)
