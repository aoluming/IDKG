# Created by ttwu at 2022/8/14
from dataclasses import dataclass
from typing import AnyStr, List


@dataclass
class MmreConfigBase:
    def __init__(self):
        self.dataset_dir: str
        self.num_labels: int = 23

        self.do_prefix: bool = True
        self.prefix_type: str = "input"  # or "full"
        self.prefix_len: int = 5

        self.classification_feat: str = "avg"  # or "pointer",
        # then use the first token after prefix for prediction

        self.check_conflict()

    def check_conflict(self):
        # check for number of classes
        assert self.num_labels > 0

        # check for prefix length
        if self.do_prefix:
            assert self.prefix_len > 0
            assert self.prefix_type in ("input", "full")




