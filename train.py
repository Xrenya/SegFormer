import torch.multiprocessing as mp
import torch
import os
from omegaconf import OmegaConf
import argparse
import importlib


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def main_worker(rank, opt, world_size):
    torch.cuda.set_device(rank)
    device = torch.device(rank)
    print(device)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    torch.distributed.init_process_group('nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    # trainer = Trainer(opt, device, rank)
    trainer = instantiate_from_config(opt)(opt, device, rank)
    trainer.iterate()
    print('train finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='./configs/segformer.yaml', help='config path')
    parser.add_argument('--train_url', type=str, default='./experiments/ckpts/tensorboard', help='train output path')
    args, unparsed = parser.parse_known_args()

    opt = OmegaConf.load(args.config)
    opt['train_url'] = args.train_url

    world_size = torch.cuda.device_count()
    print('world_size: ', world_size)
    mp.spawn(
        main_worker,
        args=(
            opt,
            world_size,
        ),
        nprocs=world_size,
        join=True
    )




