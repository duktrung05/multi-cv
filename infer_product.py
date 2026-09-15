"""Inspect one or more product images with a trained checkpoint."""
import argparse
import json
from src.domain.inspection import DEFAULT_CONFIG, DEFAULT_WEIGHTS, inspection_payload
from src.infrastructure.ml.model_factory import ModelFactory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    inspector = ModelFactory.build(args.config, args.weights, args.device)
    for path in args.image:
        print(json.dumps({"image": path, **inspection_payload(inspector.classify_image(path))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
