import argparse
import os

os.system('pip install -r requirements.txt')
os.system('python -m torch.utils.collect_env')

parser = argparse.ArgumentParser(
    prog='SemanticSegmentation',
    description='Indoor Semantic Segmentation',
    epilog='Input the yaml configuration file'
)
parser.add_argument(
    '-c',
    '--config',
    type=str,
    default="configs/segformer.yaml",
    help='config path'
)
args, unparsed = parser.parse_known_args()

command = f'python train.py --config {args.config}'
os.system(command)
