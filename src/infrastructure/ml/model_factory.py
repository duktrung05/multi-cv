"""Application adapter for the shared strict DGM4 predictor."""
from src.infrastructure.ml.dgm4 import DGM4Predictor
from src.domain.forensics import classification_result


class DGM4Classifier:
    def __init__(self, predictor):
        self.predictor = predictor
        self.class_names = predictor.class_names

    def classify_image(self, image_path, criteria=0):
        return classification_result(self.predictor.predict(image_path), criteria)


class ModelFactory:
    @staticmethod
    def build(config, checkpoint, device="auto"):
        return DGM4Classifier(DGM4Predictor.load(config, checkpoint, device))
