"""Product inspection demo using the same model and policy as the API."""
import argparse
from src.domain.inspection import DEFAULT_CONFIG, DEFAULT_WEIGHTS, inspection_payload
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
            raise gr.Error("Vui lòng chọn ảnh sản phẩm")
        result = inspector.classify_image(path)
        return result.coarse_label, result.meta["scores"], inspection_payload(result)

    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Ảnh sản phẩm"),
        outputs=[gr.Textbox(label="Kết quả PASS / REVIEW / FAIL"),
                 gr.Label(num_top_classes=len(inspector.class_names), label="Điểm dự đoán từng lỗi"),
                 gr.JSON(label="Chi tiết kiểm tra")],
        title="detect_bolt — Kiểm tra lỗi sản phẩm",
        description="Tải ảnh của loại sản phẩm đã được huấn luyện. REVIEW: cần người kiểm tra lại. "
                    "Kết quả phân loại toàn ảnh; chưa khoanh vùng lỗi.",
    ).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
