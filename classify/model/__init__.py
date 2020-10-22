
from .custom_model import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenet import *
from .mnasnet import mnasnet1_0_modified

# def build_model(cfg):
#     if cfg.MODEL.ARCH == "resnet18":
#         model = resnet18(cfg.MODEL.PRETRAINED)
#     elif cfg.MODEL.ARCH == "resnet34":
#         model = resnet34(cfg.MODEL.PRETRAINED)
#     elif cfg.MODEL.ARCH == "resnet50":
#         model = resnet50(cfg.MODEL.PRETRAINED)
#     elif cfg.MODEL.ARCH == "resnet101":
#         model = resnet101(cfg.MODEL.PRETRAINED)
#     elif cfg.MODEL.ARCH == "resnet152":
#         model = resnet152(cfg.MODEL.PRETRAINED)
#     else:
#         assert False, "Arch ==>{}<== is not found!".format(cfg.MODEL.ARCH)
#     return model
