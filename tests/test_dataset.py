import numpy as np
import pytest
from omegaconf import OmegaConf

from src.utils import instantiate_from_config


@pytest.mark.parametrize("config, input_shape, output_shape", [
    ("./configs/segformer.yaml", (650, 650, 3), (3, 512, 512)),
])
def test_ade30k_aug(config, input_shape, output_shape):
    opt = OmegaConf.load(config)
    augmentations = instantiate_from_config(
        opt.augmentations)(**opt.augmentations.params)
    test_aug = augmentations.test_transformation
    image = np.random.randint(low=0, high=255,
                              size=input_shape).astype(np.float)
    aug = test_aug(image=image)["image"]

    assert tuple(aug.shape) == output_shape, (
        f"Expected output shape={output_shape}, but got {aug.shape}")

    train_aug = augmentations.train_transformation
    aug = train_aug(image=image)["image"]

    assert tuple(aug.shape) == output_shape, (
        f"Expected output shape={output_shape}, but got {aug.shape}")


@pytest.mark.parametrize("config", [
    "./configs/segformer.yaml",
])
def test_ade30k_aug_dataset(config):
    opt = OmegaConf.load(config)
    augmentations = instantiate_from_config(
        opt.augmentations)(**opt.augmentations.params)
    train_dataset = instantiate_from_config(
        opt.dataset)(**opt.dataset.params,
                     transform=augmentations.test_transformation)
    image_size = opt.augmentations.params.image_size
    data = train_dataset[0]
    assert "image" in data, (
        "Expected key 'image' in dataset ouput, but got 'None'")
    assert "mask" in data, (
        "Expected key 'mask' in dataset ouput, but got 'None'")
    assert data["image"].shape[-2:] == (image_size, image_size), (
        f"Expected image size=({image_size, image_size}), "
        "but got {data['image'].shape[-2:]}"
    )
