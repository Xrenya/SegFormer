import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.losses.loss import Total_loss


@pytest.mark.parametrize("config, classes", [
    ("./configs/segformer.yaml", 20),
])
def test_total_loss(config, classes):
    opt = OmegaConf.load(config)
    loss_fn = Total_loss(opt.loss)
    appoximate_loss = -np.log(1 / classes)
    low, high = appoximate_loss - 0.3, appoximate_loss + 0.3
    input_tensor = torch.rand(3, classes, 512, 512)
    target_tensor = torch.randint(low=0, high=20, size=(3, 512, 512))
    loss, loss_dict = loss_fn(input_tensor, target_tensor)
    assert "loss" in loss_dict, (
        "Expected key 'loss', but got 'None'"
    )
    assert low <= loss <= high, (
        "Expected loss value should be in interval "
        f"[{low, high}], but got '{loss}'"
    )
