"""Classify images using a trained DGM4 REAL / AI_EDITED checkpoint."""
import argparse
import json
from src.infrastructure.ml.dgm4 import DEFAULT_CONFIG, DEFAULT_WEIGHTS, DGM4Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    predictor = DGM4Predictor.load(args.config, args.weights, args.device)
    for path in args.image:
        print(json.dumps({"image": path, **predictor.predict(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
