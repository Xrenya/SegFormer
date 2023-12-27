import pytest
import torch
from omegaconf import OmegaConf

from src.utils import instantiate_from_config


@pytest.mark.parametrize("config", [
    "./configs/segformer.yaml",
])
def test_train_config(config):
    opt = OmegaConf.load(config)
    assert "gpu" in opt, ("Expected key 'gpu' in config, but got 'None'")
    assert "target" in opt, ("Expected key 'target' in config, but got 'None'")


@pytest.mark.parametrize("config", [
    "./configs/segformer.yaml",
])
def test_segformer_b0_config(config):
    opt = OmegaConf.load(config)
    config = instantiate_from_config(opt.network.config)()
    assert config.depths == [
        2, 2, 2, 2
    ], (f"Expected depth={[2, 2, 2, 2]}, but got {config.depths}")
    assert config.sr_ratios == [
        8, 4, 2, 1
    ], (f"Expected sr_ratios={[8, 4, 2, 1]}, but got {config.sr_ratios}")


@pytest.mark.parametrize("config, output_shape", [
    ("./configs/segformer.yaml", (1, 150, 512, 512)),
])
def test_segformer_b0_inference(config, output_shape):
    opt = OmegaConf.load(config)
    config = instantiate_from_config(opt.network.config)()
    model = instantiate_from_config(opt.network)(config)
    input_tensor = torch.rand((1, 3) + output_shape[-2:])
    pred = model(input_tensor)
    assert pred.shape == output_shape, (
        f"Expected output shape {output_shape}, but got {pred.shape}")
