import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
# import cfg
# from tqdm import tqdm
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from src.losses.loss import Total_loss
from src.utils import get_world_size, instantiate_from_config

from .lr_scheduler import MultiStepRestartLR
from .utils import (create_ckpt_dir, del_file, fix_model_state_dict, makedirs,
                    reduce_loss_dict)

# from torchvision.utils import make_grid, save_image


# reduce_value, to_items


class Trainer(object):

    def __init__(self, opt, device, rank):
        self.opt = opt
        self.flag_epoch = 0
        self.global_step = -1  # 49990  #-1
        self.device = device
        self.rank = rank
        # define dataloader
        self.eval = False
        self.dataloader = self.make_data_loader()
        # define models
        config = instantiate_from_config(self.opt.network.config)()
        model = instantiate_from_config(self.opt.network)(config)
        self.model = self.init_net(model.to(self.device))
        # define optimizer
        self.optimizer, self.lr_scheduler = self.init_optimizer()
        # define loss
        self.loss_fn = Total_loss(self.opt.loss).to(self.device)

        if self.opt.resume_path:
            self.resume_model()

        self.loss_dict = {}
        self.tb_writer = None
        if self.rank == 0:
            (
                self.save_dir,
                self.tensorboard_dir,
                self.temp_val_dir
            ) = create_ckpt_dir(
                self.opt.ckpt_dir,
                self.opt.name
            )
            makedirs(self.opt.train_url)
            del_file(self.opt.train_url)
            self.tb_writer = SummaryWriter(self.opt.train_url)

    def iterate(self):
        if self.rank == 0:
            print('Start the training')
        for epoch in range(self.flag_epoch, self.opt.max_epoch + 1):
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            pbar = tqdm(self.dataloader["train"])
            for step, batch_data in enumerate(pbar):
                self.global_step += 1
                image = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                output = self.train_step(image, mask)

                self.lr_scheduler.step()

                # log the loss and img
                if self.rank == 0 and self.global_step % (
                    self.opt.log_interval
                ) == 0:
                    self.report(epoch)
                    self.log_loss()

                # validation
                if self.eval and self.rank == 0 and self.global_step % (
                self.opt.val_interval) == 0 and epoch >= 1:
                    # self.log_img(output, hq_img)
                    metrics = self.validation()
                    self.log_metrics(metrics, self.global_step)

                # save the model
                if (
                    self.rank == 0
                    and self.global_step % self.opt.save_model_interval == 0
                    and epoch >= 1
                ):
                    self.save_model(epoch, self.global_step)

            if self.global_step > self.opt.total_iter:
                self.save_model(epoch, self.global_step)
                break

    def validation(self):
        pbar = tqdm(self.dataloader["eval"])
        self.model.eval()
        accuracy = []
        for step, batch_data in enumerate(pbar):
            image = batch_data['image'].to(self.device)
            mask = batch_data['mask']
            
            with torch.no_grad():
                logits = self.model(image)

            logits = nn.functional.interpolate(
                logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )
            logits = F.softmax(logits, dim=1)
            logits = logits.argmax(1).detach().cpu().numpy()
            
            mask = mask.detach().cpu().numpy()
            pixel_wise_accuracy = accuracy_score(mask.flatten(), logits.flatten())

            accuracy.append(pixel_wise_accuracy)
            pbar.set_postfix({"Batch": step, "Accuracy": np.mean(pixel_wise_accuracy)})

        return {
            "accuracy": float(np.mean(accuracy))
        }


    def train_step(self, image, mask):
        self.model.train()
        output = self.model(image)

        self.optimizer.zero_grad()
        self.backward(output, mask)
        self.optimizer.step()

        self.log_dict = reduce_loss_dict(self.rank, self.loss_dict)
        return output

    def backward(self, output, mask):
        loss, loss_dict = self.loss_fn(output, mask)
        loss.backward()
        self.loss_dict['loss'] = loss
        self.loss_dict.update(loss_dict)

    # def log_img(self, output, gt, name='train'):
    #     if self.tb_writer is not None:
    #         dis_row = 4  # <= batchsize
    #         images = torch.cat(
    #           (output[0:dis_row, ...], gt[0:dis_row, ...]),
    #          0)
    #         images = images * 0.5 + 0.5
    #         grid = make_grid(images, nrow=dis_row, padding=10)
    #         self.tb_writer.add_image(name, grid, self.global_step)

    def init_net(self, model):
        if self.rank == 0:
            print(f"Loading the net in GPU:{self.rank}")
        if self.opt.network.pretrained:
            checkpoint = torch.load(
                self.opt.network.pretrained,
                map_location=self.device
            )
            model.load_state_dict(checkpoint)

        if get_world_size() < 2:
            model = nn.parallel.DataParallel(model)
        else:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(
                model
            ).to(self.device)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                find_unused_parameters=False
            )  # False=True,
        return model

    def init_optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.opt.optimizer.initial_lr_g,
            betas=self.opt.optimizer.beta
            )
        lr_scheduler = MultiStepRestartLR(
            optimizer,
            milestones=self.opt.optimizer.milestones,
            gamma=self.opt.optimizer.gamma
        )
        return optimizer, lr_scheduler

    def make_data_loader(self):
        if self.rank == 0:
            print("Loading Dataset...")

        self.augmentations = instantiate_from_config(
            self.opt.augmentations)(**self.opt.augmentations.params)

        return_dataloader = {}

        if get_world_size() < 2:
            train_dataset = instantiate_from_config(self.opt.dataset)(
                **self.opt.dataset.params,
                transform=self.augmentations.train_transformation
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.opt.batch_size,
                shuffle=True,
                pin_memory=True
            )
            self.train_sampler = None
            
        else:
            train_dataset = instantiate_from_config(
                self.opt.dataset
            )(
                **self.opt.dataset.params,
                transform=self.augmentations.train_transformation
            )
            self.train_sampler = DistributedSampler(train_dataset)
            train_batch_sampler = BatchSampler(
                self.train_sampler,
                self.opt.batch_size,
                drop_last=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )

        if "val_params" in self.opt.dataset:
            eval_dataset = instantiate_from_config(self.opt.dataset)(
                **self.opt.dataset.val_params,
                transform=self.augmentations.test_transformation
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.opt.batch_size,
                shuffle=False,
                pin_memory=True
            )
            return_dataloader["eval"] = eval_loader
            self.eval = True

        return_dataloader["train"] = train_loader

        return return_dataloader

    def report(self, epoch):
        print(
            f'[STEP: {self.global_step:>6} \
            / EPOCH: {epoch:>3} \
            | Loss: {self.loss_dict["loss"]:.4f}'
        )

    def log_loss(self):
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(
                'loss',
                self.log_dict,
                self.global_step
            )
            self.tb_writer.add_scalar(
                'LR/lr',
                self.optimizer.state_dict()['param_groups'][0]['lr'],
                self.global_step
            )
        
    
    def log_metrics(self, metrics, step):
        if self.tb_writer is not None:
            for metics_name, value in metrics.items():
                self.tb_writer.add_scalar(
                    metics_name,
                    value,
                    step
                )

    def save_model(self, epoch, note=''):
        print('Saving the model...')
        save_files = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step
        }
        torch.save(
            save_files,
            f'{self.save_dir}/{self.opt.name}_{note}.pth'
        )

    def resume_model(self):
        print("Loading the trained params and the state of optimizer...")
        checkpoint = torch.load(self.opt.resume_path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.flag_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        print(
            f"Resuming from epoch: \
            {self.flag_epoch}, \
            global step: {self.global_step}"
        )

    def load_model(self):
        checkpoint = torch.load(self.opt.model_path, map_location=self.device)
        self.model.load_state_dict(fix_model_state_dict(checkpoint['params']))
