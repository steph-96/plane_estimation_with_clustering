"""
This file implements the training of the PEC model.
"""

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim

from datetime import datetime
from Evaluation import NetworkEvaluation
from Utilities.generate_planes import GeneratePlanes
from models import Model
from settings import TrainingSettings
from Training.Dataloader import loaddata
from Training.loss import Loss
from Utilities.DatasetEnum import DatasetEnum
from Utilities.Logger import Logger, AverageMeter
from Utilities.util import Utility


def main():
    """Training the PEC model with the settings defined in the settings.py file."""
    # loading SENet model
    model = Model.Model(num_features=2048, block_channel=[256, 512, 1024, 2048])

    # Supporting 1, 2, 4, 8 GPUs
    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = TrainingSettings.batch_size_8gpu
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = TrainingSettings.batch_size_4gpu
    elif torch.cuda.device_count() == 2:
        model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
        batch_size = TrainingSettings.batch_size_2gpu
    else:
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        batch_size = TrainingSettings.batch_size_1gpu

    # If configured loading a trained model to continue training on
    if TrainingSettings.network_file is not None:
        save_path = TrainingSettings.network_file
        state_dict = torch.load(save_path)['state_dict']
        model.load_state_dict(state_dict)

    cudnn.benchmark = False
    optimizer = torch.optim.Adam(model.parameters(), TrainingSettings.lr, weight_decay=TrainingSettings.weight_decay)

    # Initialising the Tensorboard logging
    date_time = datetime.now().strftime("%m%d%H%M%S")
    train_logger = Logger(TrainingSettings.tb_folder_name+"/"+TrainingSettings.tb_run_name+"_"+date_time,
                          log_frequency=TrainingSettings.tb_log_freq)

    # Initialising the Dataloader for Hypersim or SCANet
    dataloader = loaddata.get_training_data(TrainingSettings.dataset_dir, batch_size, TrainingSettings.training_datasets)

    # If configured the network is evaluated after each epoch
    if TrainingSettings.evaluate_each_epoch:
        eval_logger = Logger(TrainingSettings.tb_val_folder_name+"/"+TrainingSettings.tb_run_name+"_"+date_time)
        test_loader = loaddata.get_testing_data_hypersim(TrainingSettings.dataset_dir, TrainingSettings.eval_batch_size)
        dataset = DatasetEnum.HYPERSIM

    train_logger.writer.add_text("Batch size", f"{len(dataloader)}")
    train_logger.writer.add_text("GPU", f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    # Training loop
    for epoch in range(TrainingSettings.start_epoch, TrainingSettings.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(dataloader, model, optimizer, epoch, train_logger)
        if TrainingSettings.evaluate_each_epoch:
            result = NetworkEvaluation.evaluate(test_loader, model, eval_logger.writer, dataset)
            NetworkEvaluation.log_depth_metric(result.get_depth_metrics(), eval_logger.writer, dataset, epoch)
            NetworkEvaluation.log_planes_metric(result.get_plane_metrics(), eval_logger.writer, dataset, epoch)
            NetworkEvaluation.log_normal_metric(result.get_normal_metrics(), eval_logger.writer, dataset, epoch)
        if epoch % 2 == 0 and epoch >= 14:
            save_checkpoint({'state_dict': model.state_dict()}, f"{date_time}_checkpoint_{epoch}.tar")


def train(dataloader, model, optimizer, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    loss_fct = Loss(logger)
    total_batch_size = len(dataloader)
    end = time.perf_counter()

    for i, sample_batched in enumerate(dataloader):
        image, gt_normals, gt_depth = sample_batched['image'], sample_batched['normal'], sample_batched['depth']
        mask, plane_information = sample_batched['mask'], sample_batched['planes']

        # Combining surface normals and plane parameter d to plane parameters.
        gt_plane_para = torch.cat((gt_normals, plane_information[:, 0:1]), dim=1).cuda()
        # Extracting the plane segmentation from the dataloader.
        gt_labels = plane_information[:, 1].cuda().type(torch.int64).unsqueeze(1)

        # Additional learning mask setting. Ignoring pixel that are further away than a threshold.
        mask[mask > TrainingSettings.max_training_depth] = 0

        # skip all empty masks
        if torch.sum(mask) == 0:
            continue
        mask = mask.cuda()

        image = image.cuda()
        gt_normals = gt_normals.cuda(non_blocking=True)
        gt_depth = gt_depth.cuda(non_blocking=True)

        image = torch.autograd.Variable(image)
        gt_normals = torch.autograd.Variable(gt_normals)
        mask = torch.autograd.Variable(mask)
        gt_plane_para = torch.autograd.Variable(gt_plane_para)
        gt_labels = torch.autograd.Variable(gt_labels)

        optimizer.zero_grad()

        gt_zeros = gt_labels.sum(dim=1).sum(dim=1).sum(dim=1) == 0
        gt_labels[gt_zeros] = 1
        mask[gt_zeros] = 0

        batch_size, _, height, width = image.size()

        output = model(image)

        loss = loss_fct.calc_loss(output, gt_normals, gt_depth, gt_labels, mask)

        losses.update(loss.item(), batch_size)
        loss.backward()
        optimizer.step()

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        logger.print_info(epoch, i, total_batch_size, batch_time)

    # Generate epoch log
    count = 0
    with torch.no_grad():
        while True:
            try:
                # take random image from last batch
                random_int = np.random.randint(0, image.size(0))
                emb = output[1][random_int]

                # Create label image with hdbscan
                label_image = Utility.embedding_segmentation(emb.unsqueeze(0)).squeeze(0)
                depth = torch.norm(output[0][random_int], dim=0, keepdim=True)
                plane_para = GeneratePlanes.plane_params_from_normals_and_depth(output[0][random_int]/depth,
                                                                                depth, DatasetEnum.HYPERSIM)
                logger.epoch_log(batch_time, image[random_int], plane_para.squeeze(), gt_plane_para[random_int],
                                 label_image, gt_labels[random_int], depth, gt_depth[random_int], epoch)
                break
            except AttributeError:
                # if something fails with that image a new one is logged
                if count > image.size(0):
                    break
                count += 1


def adjust_learning_rate(optimizer, epoch):
    """Adjusts the learning rate every 5 epochs.

    Parameters:
        optimizer (torch.optim.Adam): Optimiser used for the training
        epoch (int): Current epoch
    """
    lr = TrainingSettings.lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save the current network weighs to a file.

    Parameters:
        state (dict): Dictionary containing the current model state
        filename (str): Path including the file name where the model is saved to
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()
