"""
Implements the training loss calculation class.
"""

import torch

from torch import nn
import torch.nn.functional as F


class Loss:
    def __init__(self, logger):
        """Initialises the Loss calculating class.

        Parameters:
            logger (Utilities.Logger.Logger): Tensorboard logger instance
        """
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.logger = logger

    def calc_loss(self, output, gt_normals, gt_depth, gt_labels, mask):
        """Calculates the training loss

        Parameters:
            output (tuple): Output of the network. Tuple of two Tensors (output_param (b, 3, h, w), output_embedding (b, 8, h, w))
            gt_normals (torch.Tensor): Ground truth surface normals (b, 3, h, w)
            gt_depth (torch.Tensor): Ground truth plane labels (b, 1, h, w)
            gt_labels (torch.Tensor): Ground truth label of plane segmentation (b, 1, h, w)
            mask (torch.Tensor): Learning mask (b, 1, h, w)

        Returns:
            torch.Tensor: Batch loss
        """
        return self.__calc_embedding_loss(output[0], output[1], gt_labels, gt_depth, gt_normals, mask)

    def __calc_embedding_loss(self, output_param, output_embedding, gt_labels, gt_depth, gt_normals, mask):
        """calculates the PEC model loss. Penalising the embedding, surface normals and depth estimation.

        Parameters:
            output_param (torch.Tensor): Surface normal and depth estimation in one vector per pixel (b, 3, h, w)
            output_embedding (torch.Tensor): Embedding vector for plane segmentation (b, 8, h, w)
            gt_normals (torch.Tensor): Ground truth surface normals (b, 3, h, w)
            gt_depth (torch.Tensor): Ground truth plane labels (b, 1, h, w)
            gt_labels (torch.Tensor): Ground truth label of plane segmentation (b, 1, h, w)
            mask (torch.Tensor): Learning mask (b, 1, h, w)

        Return:
            torch.Tensor: Batch loss
        """
        batch_size = output_param.size(0)

        # calculate depth from the output params
        output_depth = torch.norm(output_param, dim=1, keepdim=True)

        # surface normal loss, penalising the angle error between gt and output
        loss_normals_angle = torch.sum(torch.abs(1 - self.cos(output_param, gt_normals + 1e-3)) * mask) / torch.sum(mask)
        # relative depth error. The absolute squared depth error is divided by the depth to stronger penalise close
        # up object
        loss_depth = torch.sum(torch.pow(gt_depth - output_depth, 2) /
                               (output_depth + torch.Tensor([1e-3]).to(output_param.device)) * mask) / torch.sum(mask)

        # the push-pull-loss is calculated for each image of the batch
        loss_push_pull = torch.zeros(1).cuda()
        for batch_id in range(batch_size):
            loss_push_pull += self.__embedding_loss(output_embedding[batch_id], gt_labels[batch_id], mask[batch_id])

        loss = loss_normals_angle + loss_push_pull + loss_depth

        # logging to tensorboard
        self.logger.update_losses(loss_push_pull.item(), loss_depth.item(), loss_normals_angle.item(), loss.item(),
                                  batch_size)

        return loss

    @staticmethod
    def __embedding_loss(embedding, gt_labels, mask, t_pull=0.2, t_push=1.9):
        """Calculates the Push-Pull-Loss.

        Original code from: https://github.com/svip-lab/PlanarReconstruction

        Parameters:
            embedding (torch.Tensor): Outputted embedding vector (8, h, w)
            gt_labels (torch.Tensor): Ground truth plane segmentation labels (1, h, w)
            mask (torch.Tensor): Learning mask (1, h, w)
            t_pull (float): Parameter of how close the embedding vectors should be to their center (Default: 0.2)
            t_push (float): Parameter of how far away each cluster should be from one another (Default: 1.9)

        Return:
            torch.Tensor: Push-Pull-Loss
        """
        c, h, w = embedding.size()

        device = embedding.device

        num_planes = gt_labels.max()
        # transform to label mask with size (num_planes, h, w)
        gt_labels = (torch.unsqueeze(gt_labels.squeeze(), -1) == torch.arange(0, num_planes + 1).to(device)).bool()
        gt_labels = torch.moveaxis(gt_labels, -1, 0)

        # calculate pull loss

        # groups the embedding together by the ground truth segmentation
        embeddings = []
        for i in range(num_planes):
            feature = torch.transpose(torch.masked_select(embedding, gt_labels[i, :, :].view(1, h, w) * mask).view(c, -1), 0, 1)
            if feature.numel() > 0:
                embeddings.append(feature)
            else:
                embeddings.append(torch.zeros((1, c)).to(device))

        # calculate centers
        centers = []
        for feature in embeddings:
            center = torch.mean(feature, dim=0).view(1, c)
            centers.append(center)

        # calculates the pull loss by adding up all embedding vectors that are too far away from the center
        pull_loss = torch.Tensor([0.0]).to(device)
        for feature, center in zip(embeddings, centers):
            dis = torch.norm(feature - center, 2, dim=1) - t_pull
            dis = F.relu(dis)
            pull_loss += torch.mean(dis)
        pull_loss /= int(num_planes)

        # can't calculate push loss with only one plane
        if num_planes == 1:
            return pull_loss

        # calculates push loss

        centers = torch.cat(centers, dim=0)
        # calculate center-center distance matrix
        a = centers.repeat(1, int(num_planes)).view(-1, c)
        b = centers.repeat(int(num_planes), 1)
        distance = torch.norm(a - b, 2, dim=1).view(int(num_planes), int(num_planes))

        # unselect same center distance
        eye = torch.eye(int(num_planes)).to(device)
        pair_distance = torch.masked_select(distance, eye == 0)

        # penalise all center distance that are too close to one another
        pair_distance = t_push - pair_distance
        pair_distance = F.relu(pair_distance)
        push_loss = torch.mean(pair_distance).view(-1)

        return pull_loss + push_loss
