import os
import torch
# import cfg
# from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import to_items, reduce_value, reduce_loss_dict, fix_model_state_dict, del_file
from torchvision.utils import make_grid, save_image
from .utils import create_ckpt_dir, makedirs
from torch.utils.tensorboard import SummaryWriter

from src.losses.loss import Total_loss
from .lr_scheduler import MultiStepRestartLR
from src.utils import instantiate_from_config, get_world_size


class Trainer(object):

    def __init__(self, opt, device, rank):
        self.opt = opt
        self.flag_epoch = 0
        self.global_step = -1  #49990  #-1
        self.device = device
        self.rank = rank
        ########### define dataloader ##########################
        self.train_loader = self.make_data_loader()
        ########### define models #################
        config = instantiate_from_config(self.opt.network.config)()
        model = instantiate_from_config(self.opt.network)(config)
        self.model = self.init_net(model.to(self.device))
        ########### define optimizer #######################
        self.optimizer, self.lr_scheduler = self.init_optimizer()
        ########### define loss ###################
        self.loss_fn = Total_loss(self.opt.loss).to(self.device)

        if self.opt.resume_path:
            self.resume_model()

        self.loss_dict = {}
        self.tb_writer = None
        if self.rank == 0:
            self.save_dir, self.tensorboard_dir, self.temp_val_dir = create_ckpt_dir(self.opt.ckpt_dir, self.opt.name)
            makedirs(self.opt.train_url)
            del_file(self.opt.train_url)
            self.tb_writer = SummaryWriter(self.opt.train_url)

    def iterate(self):
        if self.rank == 0:
            print('Start the training')
        for epoch in range(self.flag_epoch, self.opt.max_epoch + 1):
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            for step, batch_data in enumerate(self.train_loader):
                self.global_step += 1
                image = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                output = self.train_step(image, mask)

                self.lr_scheduler.step()

                # log the loss and img
                if self.rank == 0 and self.global_step % (self.opt.log_interval) == 0:
                    self.report(epoch)
                    self.log_loss()

                # validation
                # if self.rank == 0 and self.global_step % (self.opt.val_interval) == 0 and epoch >= 1:
                #     self.log_img(output, hq_img)
                #     self.validation()

                # save the model
                if self.rank == 0 and self.global_step % self.opt.save_model_interval == 0 and epoch >= 1:
                    self.save_model(epoch, self.global_step)

            if self.global_step > self.opt.total_iter:
                self.save_model(epoch, self.global_step)
                break

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
    #         images = torch.cat((output[0:dis_row, ...], gt[0:dis_row, ...]), 0)
    #         images = images * 0.5 + 0.5
    #         grid = make_grid(images, nrow=dis_row, padding=10)
    #         self.tb_writer.add_image(name, grid, self.global_step)

    def init_net(self, model):
        if self.rank == 0:
            print(f"Loading the net in GPU")
        if self.opt.network.pretrained:
            checkpoint = torch.load(self.opt.network.pretrained, map_location=self.device)
            model.load_state_dict(checkpoint)
        
        if get_world_size() < 2:
            model = torch.nn.parallel.DataParallel(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                find_unused_parameters=False
            )  #False=True,
        return model

    def init_optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.opt.optimizer.initial_lr_g, betas=self.opt.optimizer.beta)
        lr_scheduler = MultiStepRestartLR(
            optimizer, milestones=self.opt.optimizer.milestones, gamma=self.opt.optimizer.gamma)
        return optimizer, lr_scheduler

    def make_data_loader(self):
        if self.rank == 0:
            print("Loading Dataset...")

        self.augmentations = instantiate_from_config(self.opt.augmentations)(**self.opt.augmentations.params)

        if get_world_size() < 2:
            train_dataset = instantiate_from_config(self.opt.dataset)(
                **self.opt.dataset.params,
                transform=self.augmentations.train_transformation
            )
            train_loader = DataLoader(
                dataset=train_dataset, batch_size=self.opt.batch_size, shuffle=True, pin_memory=True)
            self.train_sampler = None
        else:
            train_dataset = instantiate_from_config(self.opt.dataset)(
                **self.opt.dataset.params,
                transform=self.augmentations.train_transformation
            )
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_batch_sampler = torch.utils.data.BatchSampler(self.train_sampler, self.opt.batch_size, drop_last=True)
            train_loader = DataLoader(
                train_dataset, batch_sampler=train_batch_sampler, shuffle=False, num_workers=8, pin_memory=True)
            # if self.opt.dataset.val_data_root:
            #     self.val_dataset = DatasetFromFolder(self.opt.dataset.val_data_root, is_train=False)
        return train_loader

    def report(self, epoch):
        print(
            f'[STEP: {self.global_step:>6} / EPOCH: {epoch:>3} | Loss: {self.loss_dict["loss"]:.4f}'
        )

    def log_loss(self):
        if self.tb_writer is not None:
            self.tb_writer.add_scalars('loss', self.log_dict, self.global_step)
            self.tb_writer.add_scalar('LR/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.global_step)

    def save_model(self, epoch, note=''):
        print('Saving the model...')
        save_files = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step
        }
        torch.save(save_files, f'{self.save_dir}/{self.opt.name}_{note}.pth')

    def resume_model(self):
        print("Loading the trained params and the state of optimizer...")
        checkpoint = torch.load(self.opt.resume_path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.flag_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        print(f"Resuming from epoch: {self.flag_epoch}, global step: {self.global_step}")

    def load_model(self):
        checkpoint = torch.load(self.opt.model_path, map_location=self.device)
        self.model.load_state_dict(fix_model_state_dict(checkpoint['params']))
    #
    # def validation(self):
    #     print('Start the validation...')
    #     import cv2
    #     import os
    #     import glob
    #     import torchvision.transforms as transforms
    #     from PIL import Image
    #     import numpy as np
    #
    #     def tensor2img(tensor):
    #         tensor = tensor.detach().clamp_(-1, 1)  #.squeeze(0)
    #         tensor = (tensor * 0.5 + 0.5).mul(255).byte()
    #         # jpg_img = transforms.ToPILImage()(tensor)
    #         npimg = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
    #         cvimg = cv2.cvtColor(np.asarray(npimg), cv2.COLOR_RGB2BGR)
    #         return cvimg
    #
    #     def patch2img(p_list):
    #         output = np.zeros((1024, 1024, 3), np.uint8)
    #         uint = 1024 // 8
    #         output[0:uint * 3, :uint * 3, :] = p_list[0][0:uint * 3, :uint * 3, :]
    #         output[0:uint * 3, uint * 3:uint * 5, :] = p_list[1][0:uint * 3, uint * 1:uint * 3, :]
    #         output[0:uint * 3, uint * 5:uint * 8, :] = p_list[2][0:uint * 3, uint * 1:, :]
    #         output[uint * 3:uint * 5, :uint * 3, :] = p_list[3][uint * 1:uint * 3, :uint * 3, :]
    #         output[uint * 3:uint * 5, uint * 3:uint * 5, :] = p_list[4][uint * 1:uint * 3, uint * 1:uint * 3, :]
    #         output[uint * 3:uint * 5, uint * 5:uint * 8, :] = p_list[5][uint * 1:uint * 3, uint * 1:, :]
    #         output[uint * 5:, :uint * 3, :] = p_list[6][uint * 1:, :uint * 3, :]
    #         output[uint * 5:, uint * 3:uint * 5, :] = p_list[7][uint * 1:, uint * 1:uint * 3, :]
    #         output[uint * 5:, uint * 5:uint * 8, :] = p_list[8][uint * 1:, uint * 1:, :]
    #         # for i in range(9):
    #         # cv2.imshow('output', output)
    #         # cv2.waitKey()
    #         return output
    #
    #     def img2patch(img, psize=512):
    #         patch_list = []
    #         H, W, C = img.shape
    #         for i in range(3):
    #             for j in range(3):
    #                 x1, y1 = j * psize // 2, i * psize // 2
    #                 x2, y2 = x1 + psize, y1 + psize
    #                 patch = img[y1:y2, x1:x2, :]
    #                 patch_list.append(patch)
    #         return patch_list
    #
    #     # root = 'C:/Users/t50030225/Downloads/images_resize_512x512'
    #
    #     self.net_G.eval()
    #     del_file(self.temp_val_dir)
    #     self.net_G.module.quantizer.reset_usage()
    #
    #     if 'patch' in self.opt.name:
    #         val_list = glob.glob(f"{self.opt.dataset.gt_val_data_root}/*")
    #
    #         transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    #         transform = transforms.Compose(transform_list)
    #
    #         for img_path in val_list[:]:
    #             # print(img_path)
    #             img_name = os.path.split(img_path)[-1]
    #             img = cv2.imread(img_path)
    #             img = cv2.resize(img, (1024, 1024), cv2.INTER_CUBIC)
    #             patch_list = img2patch(img)
    #             patch_len = len(patch_list)
    #             batch_size = 4  # should not over 9
    #
    #             out_list = []
    #             for iter in range(patch_len // batch_size + 1):
    #                 batch_list = patch_list[iter * batch_size:iter * batch_size + batch_size]
    #
    #                 patch_img = Image.fromarray(cv2.cvtColor(batch_list[0], cv2.COLOR_BGR2RGB))
    #                 patch_tensor = transform(patch_img).unsqueeze(0)
    #
    #                 for patch_img in batch_list[1:]:
    #                     patch_img = Image.fromarray(cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB))
    #                     patch_tensor = torch.cat((patch_tensor, transform(patch_img).unsqueeze(0)), 0)
    #                 # print(patch_tensor.shape)
    #
    #                 with torch.no_grad():
    #                     outputs, _, = self.net_G(patch_tensor)
    #                     for img_tensor in outputs:
    #                         patch_cv = tensor2img(img_tensor)
    #                         out_list.append(patch_cv)
    #
    #             img_cv = patch2img(out_list)
    #             temp_path = f'{self.temp_val_dir}/{img_name}'.replace('.jpg', '.png')
    #             cv2.imwrite(temp_path, img_cv)
    #     else:
    #         from src.datasets.dataset import DatasetFromFolder
    #         self.val_dataset = DatasetFromFolder(self.opt.dataset.gt_val_data_root, is_train=False)
    #         val_loader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=8)
    #         del_file(self.temp_val_dir)
    #         for step, batch_data in enumerate(val_loader):
    #             hq_imgs, data_paths = batch_data['gt'].to(self.device), batch_data['gt_path']
    #             with torch.no_grad():
    #                 outputs, _, = self.net_G(hq_imgs)
    #                 for img_tensor, img_path in zip(outputs, data_paths):
    #                     img_cv = tensor2img(img_tensor)
    #                     img_name = os.path.split(img_path)[-1]
    #                     temp_path = f'{self.temp_val_dir}/{img_name}'
    #                     cv2.imwrite(temp_path, img_cv)
    #
    #     codebook_usage = self.net_G.module.quantizer.get_usage()
    #
    #     x = self.net_G.module.quantizer.usage.cpu().numpy()
    #     import matplotlib.pyplot as plt
    #     usage_bar = plt.figure()
    #     plt.title(f'usage: {codebook_usage*100}%')
    #     plt.bar(range(x.shape[0]), x)
    #     self.tb_writer.add_figure('usage_bar', usage_bar, self.global_step)
    #
    #     self.net_G.module.quantizer.reset_usage()
    #     print(f'{codebook_usage}')
    #
    #     device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    #     pred_folder = self.temp_val_dir
    #     gt_folder = self.opt.dataset.gt_val_data_root
    #     fid = calculate_fid(paths=[pred_folder, gt_folder], batch_size=40, device=device, dims=2048, num_workers=8)
    #     psnr, ssim = calculate_psnr_ssim(pred_folder, gt_folder)
    #     # niqe = calculate_niqe_folder(pred_folder)
    #     lpips = calculate_lpips(pred_folder, gt_folder)
    #     # idd = calculate_cos_dist(pred_folder, gt_folder)
    #
    #     print(f'Loss: {loss:.6f}')
    #     if self.tb_writer is not None:
    #         self.tb_writer.add_scalar('val/loss', psnr, self.global_step)

        # del_file(self.temp_val_dir)

