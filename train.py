"""Train or evaluate the DGM4 REAL / AI_EDITED classifier."""
import argparse
from src.infrastructure.ml.dgm4 import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG)
    parser.add_argument('-u', '--update', nargs='+', default=[], help='Config overrides: key=value')
    args = parser.parse_args()
    from engine.solver.dgm4_solver import run
    run(args.config, args.update)


if __name__ == '__main__':
    main()
