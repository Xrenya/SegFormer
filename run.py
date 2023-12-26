import shutil
import moxing as mox
import os
import argparse


os.system('pip install -r requirement.txt')
os.system('python -m torch.utils.collect_env')

command = f'python train.py --config {args.config}'
os.system(command)
