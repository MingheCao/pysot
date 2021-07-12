import torch.nn as nn
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder


class SiamModel(ModelBuilder):
    def __init__(self):
        super(SiamModel, self).__init__()

    def template_rb(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        return zf

    def corr(self, x, zf):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)
        return {
            'cls': cls,
            'loc': loc,
        }