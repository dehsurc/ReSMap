"""Vector mAP of a saved submission_vector.json.

    python tools/eval_submission.py CONFIG work_dirs/.../submission_vector.json
"""
import argparse
import os
import sys

from mmcv import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('submission')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    sys.path.append(os.path.abspath('.'))
    from plugin.datasets.evaluation.vector_eval import VectorEvaluate
    print(VectorEvaluate(cfg.data.test.eval_config).evaluate(args.submission))


if __name__ == '__main__':
    main()
