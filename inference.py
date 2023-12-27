import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf

from src.constants import PALETTE
from src.utils import instantiate_from_config

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='./configs/segformer.yaml',
                        help='Configuration file')
    parser.add_argument('-w',
                        '--weight',
                        type=str,
                        default='./experiments/pretrained/pytorch_model.bin',
                        help='Pretrained model\'s weight')
    parser.add_argument('-i',
                        '--input_path',
                        type=str,
                        default='./data/ADEChallengeData2016/images/training',
                        help='Input image')
    parser.add_argument('-o',
                        '--output_path',
                        type=str,
                        default="./output",
                        help='Output folder')
    parser.add_argument('--color',
                        type=bool,
                        default=True,
                        help='Output images in color or according to classes')
    parser.add_argument('-s',
                        '--image_size',
                        default=512,
                        type=int,
                        help='Input image size')
    parser.add_argument('--ext',
                        default='jpg',
                        type=str,
                        help='Image file extension')
    args = parser.parse_args()

    output_path = args.output_path

    opt = OmegaConf.load(args.config)
    config = instantiate_from_config(opt.network.config)()
    model = instantiate_from_config(opt.network)(config).to(device)

    weight = torch.load(args.weight)
    model.load_state_dict(weight)

    augmentations = A.Compose([
        A.LongestMaxSize(max_size=args.image_size),
        A.PadIfNeeded(
            min_height=args.image_size,
            min_width=args.image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=255,
        ),
        A.Normalize(),
        ToTensorV2(),
    ])

    images = Path(args.input_path).glob(f'**/*.{args.ext}')

    for image_path in images:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = augmentations(image=image)["image"]
        tensor_image = augmented.unsqueeze(0).to(device)

        logits = model(tensor_image)
        logits = F.softmax(logits, dim=1).squeeze(0)
        logits = logits.argmax(0).detach().cpu().numpy()

        if args.color:
            segmentation_mask = np.zeros((args.image_size, args.image_size, 3))
            for label, color in enumerate(PALETTE):
                mask = logits == label
                segmentation_mask[mask] = color
        else:
            segmentation_mask = logits

        filename = os.path.join(output_path, image_path.name)
        cv2.imwrite(filename, segmentation_mask)
