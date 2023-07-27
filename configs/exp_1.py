# Created by ttwu at 2022/8/14
from .config_base import MmreConfigBase


class Exp1(MmreConfigBase):
    def __init__(self):
        super(Exp1, self).__init__()
        self.classification_feat = "pointer"

