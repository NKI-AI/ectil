from pathlib import Path
from typing import Optional

import torch
import torchvision
from torch import jit, nn

from ectil.utils import pylogger

log = pylogger.get_pylogger(__name__)


class Encoder(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        """
        expecting x to be (batch_size * channels * height * width)
        """
        return self.net(x)


class RetCCL(Encoder):
    """
    Loads the architecture and weights of Wang et al.
    https://github.com/Xiyue-Wang/RetCCL
    @article{WANG2023102645,
    title = {RetCCL: Clustering-guided contrastive learning for whole-slide image retrieval},
    author = {Xiyue Wang and Yuexi Du and Sen Yang and Jun Zhang and Minghui Wang and Jing Zhang and Wei Yang and Junzhou Huang and Xiao Han},
    journal = {Medical Image Analysis},
    volume = {83},
    pages = {102645},
    year = {2023},
    issn = {1361-8415}
    }
    """

    def __init__(
        self,
        project_root_dir: str,
        weights_path: str = "model_zoo/retccl/retccl_best_ckpt.pth",
    ) -> None:
        # The RetCCL github repo sets
        # model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
        # All these arguments influence the .fc layer, which we set to be nn.Identity(), so they don't matter
        super().__init__(net=torchvision.models.resnet50(weights=None))
        self.net.fc = nn.Identity()
        log.info(f"Loading weights from {weights_path}")
        weights = torch.load(
            Path(project_root_dir) / weights_path,
            map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
        )
        missing, unexpected = self.net.load_state_dict(weights)
        if len(missing) > 0 or len(unexpected) > 0:
            raise ValueError(f"Missing keys: {missing}. Unexpected keys: {unexpected}")
