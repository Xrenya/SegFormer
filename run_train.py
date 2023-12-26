import shutil
import moxing as mox
import os
import argparse


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    model_name = 're-cf'
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='./configs/patch_f32_s1.yaml', help='config path')
    parser.add_argument('--train_url', type=str, default='./experiments/ckpts/tensorboard', help='train output path')
    args, unparsed = parser.parse_known_args()
    # args = parser.parse_args()

    ######## copy data #######
    s3_data_path = 's3://japan-sh/j00412222/0.Data/1.FaceEnhancement/1.Huawei/jia8k/selected_clear_jpg_crop2_resize/'
    work_data_path = '/cache/_data/jia8k/images_1024'
    makedirs(work_data_path)
    mox.file.copy_parallel(s3_data_path, work_data_path)
    s3_data_path = 's3://japan-sh/t50030225/_data/0.Public/FFHQ/images_resize_512x512'
    work_data_path = '/cache/_data/FFHQ/images_resize_512x512'
    makedirs(work_data_path)
    mox.file.copy_parallel(s3_data_path, work_data_path)
    s3_data_path = 's3://japan-sh/t50030225/_data/val_set'
    work_data_path = '/cache/_data/val_set'  # celeba_512_validation
    makedirs(work_data_path)
    mox.file.copy_parallel(s3_data_path, work_data_path)

    # os.system('nvidia-smi')
    ######## copy code #######
    s3_code_path = f's3://japan-sh/t50030225/_code/{model_name}'
    work_code_path = f'/home/ma-user/modelarts/user-job-dir/{model_name}'
    # mox.file.copy_parallel(s3_code_path, work_code_path)

    mox.file.copy_parallel(
        f's3://japan-sh/t50030225/_code/{model_name}/experiments/pretrained_models/pt_inception-2015-12-05-6726825d.pth',
        '/home/ma-user/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth')
    mox.file.copy_parallel(
        f's3://japan-sh/t50030225/_code/{model_name}/experiments/pretrained_models/vgg16-397923af.pth',
        '/home/ma-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth')

    makedirs(f'/home/ma-user/modelarts/user-job-dir/{model_name}/experiments/pretrained_models')
    makedirs(f'/home/ma-user/modelarts/user-job-dir/{model_name}/experiments/ckpts')
    mox.file.copy_parallel(
        f's3://japan-sh/t50030225/_code/{model_name}/experiments/pretrained_models/pt_inception-2015-12-05-6726825d.pth',
        f'/home/ma-user/modelarts/user-job-dir/{model_name}/experiments/pretrained_models/pt_inception-2015-12-05-6726825d.pth')
    mox.file.copy_parallel(
        f's3://japan-sh/t50030225/_code/{model_name}/experiments/pretrained_models/vgg16-397923af.pth',
        f'/home/ma-user/modelarts/user-job-dir/{model_name}/experiments/pretrained_models/vgg16-397923af.pth')

    s3_code_path = 's3://japan-sh/t50030225/_code/re-cf/experiments/pretrained_models'
    work_code_path = f'/home/ma-user/modelarts/user-job-dir/{model_name}/experiments/pretrained_models'
    mox.file.copy_parallel(s3_code_path, work_code_path)

    s3_code_path = 's3://japan-sh/t50030225/_code/re-cf/experiments/pretrained_models/lpips'
    work_code_path = f'/home/ma-user/modelarts/user-job-dir/{model_name}/experiments/pretrained_models/lpips'
    mox.file.copy_parallel(s3_code_path, work_code_path)


    work_code_path = f'/home/ma-user/modelarts/user-job-dir/{model_name}'

# /home/ma-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth
    ######## install some pacages #######
    os.chdir(work_code_path)
    os.system('pip install -r requirement.txt')
    ######## run python #######

    command = f'python train.py --config {args.config} --train_url {args.train_url}'
    os.system(command)
    print('train finished')

    # copy result and log to s3
    # import time
    # timestr = time.strftime("%m%d_%H%M")

    work_result_path = work_code_path + '/experiments/ckpts'
    s3_result_path = f's3://japan-sh/s00628293/03_FaceDE/results/'
    makedirs(s3_result_path)
    mox.file.copy_parallel(work_result_path, s3_result_path)




