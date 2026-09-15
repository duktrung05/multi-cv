#!/usr/bin/env python3
"""Gradio app: upload an image, pick classes, view Grad-CAM heatmaps side by side.

Loads the model once at startup (same loader as ``infer_vas.py``) and reuses the
Grad-CAM hook/overlay logic from ``gradcam.py``.

Usage
-----
    python gradio_gradcam_app.py
    python gradio_gradcam_app.py --config configs/... --weights outputs/.../best.pth --port 7879
"""

from __future__ import annotations

import argparse
import os
import tempfile

import gradio as gr
from PIL import Image

from gradcam import _compute_gradcam, _overlay
from infer_vas import DEFAULT_CONFIG, DEFAULT_WEIGHTS, load_image_tensor, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio Grad-CAM viewer for the IAS multilabel classifier")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (defaults to config device)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7879)
    args = parser.parse_args()

    loaded = load_model(config=args.config, weights=args.weights, device=args.device)
    model = loaded.model
    class_list = loaded.class_list
    print(f"[gradio_gradcam_app] model ready on {loaded.device}, classes={class_list}")

    def run(image: Image.Image, classes: list[str], scale: str, alpha: float):
        if image is None:
            raise gr.Error("Upload an image first")
        if not classes:
            raise gr.Error("Pick at least one class")

        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="gradcam_")
        try:
            with os.fdopen(fd, "wb") as f:
                image.convert("RGB").save(f, format="JPEG")

            tensor = load_image_tensor(tmp_path, loaded.transform).unsqueeze(0).to(loaded.device)

            gallery = []
            scores = {}
            for cls in classes:
                idx = class_list.index(cls)
                cam, prob = _compute_gradcam(model, tensor, idx, scale)
                overlay = _overlay(tmp_path, cam, loaded.size, alpha=alpha)
                gallery.append((overlay, f"{cls} (p={prob:.3f})"))
                scores[cls] = prob
            return gallery, scores
        finally:
            os.remove(tmp_path)

    with gr.Blocks(title="IAS Grad-CAM Viewer") as demo:
        gr.Markdown(
            f"## Grad-CAM viewer\nModel: `{args.config}` | Weights: `{args.weights}` | Device: `{loaded.device}`\n\n"
            "Upload an image, pick one or more classes, and compare where the model looks for each."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Image")
                classes_in = gr.CheckboxGroup(
                    choices=class_list,
                    value=[c for c in ["MALE_SEXUAL", "FEMALE", "FEMALE_GENITALIA"] if c in class_list],
                    label="Target classes",
                )
                scale_in = gr.Radio(
                    choices=["combined", "c2", "c3", "c4"],
                    value="combined",
                    label="Backbone scale ('combined' = c2+c3+c4 average, matches what the head sees; "
                    "c2=finest/1:8, c3=1:16, c4=coarsest/1:32)",
                )
                alpha_in = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Heatmap opacity")
                run_btn = gr.Button("Run Grad-CAM", variant="primary")
            with gr.Column(scale=2):
                gallery_out = gr.Gallery(label="Heatmaps", columns=3, height="auto")
                scores_out = gr.Label(label="Sigmoid confidence per selected class")

        run_btn.click(
            fn=run,
            inputs=[image_in, classes_in, scale_in, alpha_in],
            outputs=[gallery_out, scores_out],
        )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
