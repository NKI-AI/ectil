from typing import Optional

import torch
from torch import nn


class MeanMIL(nn.Module):
    def __init__(
        self,
        classifier: nn.Sequential,
        post_encoder: nn.Sequential,
        attention: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.post_encoder = post_encoder
        self.classifier = classifier
        self.attention = attention

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input features of dimension batch_size, bag_size, hidden_dimension

        Returns:

        """

        x = self.post_encoder(x)

        attention_weights = None if not self.attention else self.attention(x)
        # attention_weights: batch_size * bag_size * 1

        out_per_instance = self.classifier(x)
        # out_per_instance: batch_size * bag_size * 1

        out = (
            (attention_weights * out_per_instance).sum(dim=1)
            if attention_weights is not None
            else out_per_instance.mean(dim=1)
        )
        # Mean the instance outputs for each bag
        return {
            "out": out,
            "meta": {
                "out_per_instance": out_per_instance,
                "attention_weights": attention_weights,
            },
        }
