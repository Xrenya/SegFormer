import os
import torch
import torch.distributed as dist
import time
from collections import OrderedDict


def instantiate_from_config(config):
    import importlib
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def get_world_size():
    return dist.get_world_size()


def reduce_loss_dict(rank, loss_dict):
    """reduce loss dict.

    In distributed training, it averages the losses among different GPUs .

    Args:
        loss_dict (OrderedDict): Loss dict.
    """
    with torch.no_grad():
        world_size = get_world_size()
        keys = []
        losses = []
        for name, value in loss_dict.items():
            keys.append(name)
            losses.append(value)
        losses = torch.stack(losses, 0)
        torch.distributed.reduce(losses, dst=0)
        if rank == 0:
            losses /= world_size
        loss_dict = {key: loss for key, loss in zip(keys, losses)}

        log_dict = OrderedDict()
        for name, value in loss_dict.items():
            log_dict[name] = value.mean().item()

        return log_dict


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


def init_distributed_mode():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ['LOCAL_RANK'])
    print(rank, world_size, gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=4, rank=rank)
    dist.barrier()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_ckpt_dir(ckpt_dir, name):
    temp_val_dir = os.path.join(ckpt_dir, 'temp_val')
    makedirs(temp_val_dir)
    timestr = time.strftime('%m%d_%H%M')
    ckpt_dir = f'{ckpt_dir}/{name}_{timestr}'
    tensorboard_dir = os.path.join(ckpt_dir, 'tensorboard_logs')
    save_dir = os.path.join(ckpt_dir, 'saved_models')
    makedirs(save_dir)
    makedirs(tensorboard_dir)
    return save_dir, tensorboard_dir, temp_val_dir


def to_items(dic):
    return dict(map(_to_item, dic.items()))


def _to_item(item):
    return item[0], item[1].item()


def del_file(path_data):
    for i in os.listdir(path_data):
        file_data = os.path.join(path_data, i)
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            del_file(file_data)


def fix_model_state_dict(state_dict, del_str='module.'):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith(del_str):
            name = name[len(del_str):]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict
