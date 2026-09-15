"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .multilabel_solver import MultilabelClasSolver



from typing import Dict

TASKS :Dict[str, BaseSolver] = {
    'classification': ClasSolver,
    'multilabel_classification': MultilabelClasSolver,
    'detection': DetSolver,
}

# Import criterions to register them in GLOBAL_CONFIG
from .classification_criterion import *
from .multilabel_criterion import *
