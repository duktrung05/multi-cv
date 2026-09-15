from src.infrastructure.ml.model_factory import ProductInspector


class ProductInspectorAdapter:
    def __init__(self, classifier: ProductInspector):
        self._classifier = classifier

    def classify_image(self, image_path: str, criteria: int = 0):
        return self._classifier.classify_image(image_path, criteria=criteria)

    def classify_video(self, video_path: str, criteria: int = 0):
        raise NotImplementedError("Video inspection uses the frame extraction stage")
