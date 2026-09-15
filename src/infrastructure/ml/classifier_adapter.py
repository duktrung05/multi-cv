from src.infrastructure.ml.model_factory import DGM4Classifier


class DGM4ClassifierAdapter:
    def __init__(self, classifier: DGM4Classifier):
        self._classifier = classifier

    def classify_image(self, image_path: str, criteria: int = 0):
        return self._classifier.classify_image(image_path, criteria=criteria)

    def classify_video(self, video_path: str, criteria: int = 0):
        raise NotImplementedError("Video classification uses the frame extraction stage")
