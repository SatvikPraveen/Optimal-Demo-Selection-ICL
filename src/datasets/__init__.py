"""
Dataset loading, task templates and the task registry.
"""

from .load_agnews import load_agnews
from .load_csqa import load_commonsense_qa
from .load_sst5 import load_sst5
from .tasks import TASKS, Task, get_task, load_split, load_train_without_holdout

__all__ = [
    "load_sst5",
    "load_agnews",
    "load_commonsense_qa",
    "Task",
    "TASKS",
    "get_task",
    "load_split",
    "load_train_without_holdout",
]
