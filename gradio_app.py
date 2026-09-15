"""DGM4 face manipulation demo using the same predictor as the API."""
import argparse
from src.domain.forensics import DEFAULT_CONFIG, DEFAULT_WEIGHTS, classification_payload
from src.infrastructure.ml.model_factory import ModelFactory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7877)
    args = parser.parse_args()
    import gradio as gr
    inspector = ModelFactory.build(args.config, args.weights, args.device)

    def predict(path):
        if not path:
            raise gr.Error("Vui lòng chọn ảnh")
        result = inspector.classify_image(path)
        return result.coarse_label, result.meta["scores"], classification_payload(result)

    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Ảnh cần kiểm tra"),
        outputs=[gr.Textbox(label="Kết quả REAL / AI_EDITED"),
                 gr.Label(num_top_classes=len(inspector.class_names), label="Điểm dự đoán từng lớp"),
                 gr.JSON(label="Chi tiết dự đoán")],
        title="ReaS IAS VAS — Phát hiện khuôn mặt bị AI chỉnh sửa",
        description="Phân biệt ảnh gốc và ảnh chỉnh sửa khuôn mặt bằng AI. "
                    "Model học từ DGM4 (SimSwap/StyleCLIP); chưa khoanh vùng chỉnh sửa. "
                    "Điểm dự đoán không phải bằng chứng xác thực ảnh.",
    ).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
