"""
This class uses code copied and adapted from the PlaneRCNN paper. https://github.com/NVlabs/planercnn
"""

import numpy as np
import torch

from sklearn import metrics

from Utilities.AverageMeters import AverageMeter, RecallMeter


class Metrics:
    """This class includes all functions to evaluate the network. It keeps track of all intermediate results.

    To read out the calculated result a range of function can be used which returns all or a sub part of the following
    result dictionary:

    {"plane_metrics": {"IOU": *, "RI": *, "SC": *, "VOI": *, "perc_found_planes": *, "perc_found_pixels": *},
     "estimation_metrics": {"depth": {"REL": *, "RMSE": *, "LOG10": *, "delta1": *, "delta2": *, "delta3": *},
                            "normals": {"angle_accuracy": *, "RMSE": *}}}
    """
    def __init__(self):
        """Initialises all the AverageMeter instances for keeping track of the evaluation metrics."""
        # depth estimation
        self.__rel = AverageMeter()
        self.__rms = AverageMeter()
        self.__log10 = AverageMeter()
        self.__delta1 = AverageMeter()
        self.__delta2 = AverageMeter()
        self.__delta3 = AverageMeter()

        # normal estimation
        self.__angle_accuracy = RecallMeter()
        self.__angle_rms = AverageMeter()

        # Segmentation Metrics
        self.__iou = AverageMeter()
        self.__ri = AverageMeter()
        self.__sc = AverageMeter()
        self.__voi = AverageMeter()

        # Plane accuracy
        self.__overlapping_planes = RecallMeter()
        self.__overlapping_pixels = RecallMeter()

        # settings
        self.__iou_threshold = 0.5
        self.__plane_depth_threshold = list(np.linspace(0, 0.9, 10))
        self.__plane_normal_threshold = list(np.array([5, 11.25, 22.5, 30]))

    def evaluate_output(self, labels, gt_labels, depth, gt_depth, normals, gt_normals, mask):
        """Complete network output evaluation calculation.

        Calculates the following metrics:
        - Rel. depth error
        - Root mean square depth error
        - Log10 depth error
        - delta1,2,3 for the depth estimation
        - Angle accuracy
        - Root mean square angle error
        - IOU, RI, SC, VOI segmentation metrics
        - Plane and Pixel Recall

        Parameters:
            labels (torch.Tensor): output plane segmentation (b, 1, h, w)
            gt_labels (torch.Tensor): ground truth plane segmentation (b, 1, h, w)
            depth (torch.Tensor): depth map output (b, 1, h, w)
            gt_depth (torch.Tensor): ground truth depth map (b, 1, h, w)
            normals (torch.Tensor): output normals (b, 3, h, w)
            gt_normals (torch.Tensor): ground truth surface normals (b, 3, h, w)
            mask (torch.Tensor): data mask (b, 1, h, w)
        """
        # Check and equalise dimensions
        assert labels.dim() in (3, 4)

        if labels.dim() == 3:
            labels = labels.unsqueeze(0)
            gt_labels = gt_labels.unsqueeze(0)
            depth = depth.unsqueeze(0)
            gt_depth = gt_depth.unsqueeze(0)
            normals = normals.unsqueeze(0)
            gt_normals = gt_normals.unsqueeze(0)
            mask = mask.unsqueeze(0)

        # Calculate depth and surface normal metrics.
        self.__evaluate_normal_estimation(normals, gt_normals, mask)
        self.__evaluate_depth_estimation(depth, gt_depth, mask)

        batch_size = labels.size(0)
        device = labels.device
        mask = mask.squeeze(1).unsqueeze(-1).unsqueeze(-1)

        for batch_id in range(batch_size):
            # Create Plane label stack with each layer only containing one plane
            plane_labels_stack = (torch.unsqueeze(labels[batch_id].squeeze(), -1) == torch.arange(0, labels[batch_id].max() + 1).to(device)).float()
            gt_plane_labels_stack = (torch.unsqueeze(gt_labels[batch_id].squeeze(), -1) == torch.arange(0, gt_labels[batch_id].max() + 1).to(device)).float()
            mask_batch = mask[batch_id]
            if mask_batch.sum() == 0 or gt_plane_labels_stack.size(2) == 0 or plane_labels_stack.size(2) == 0:
                continue

            # calculate intersection and union for IOU and other metrics
            intersection_mask = gt_plane_labels_stack.unsqueeze(-1) * plane_labels_stack.unsqueeze(2) * mask_batch > 0.5
            intersection = intersection_mask.float().sum(0).sum(0)
            union = (torch.max(gt_plane_labels_stack.unsqueeze(-1), plane_labels_stack.unsqueeze(2)) * mask_batch).sum(0).sum(0).float()

            # Calculate segmentation metrics
            self.__iou_metric(intersection, union)
            self.__rand_index_metric(labels[batch_id], gt_labels[batch_id], mask[batch_id])
            self.__voi_metric(intersection)

            # Check if there is more than one plane
            if 1 in gt_plane_labels_stack.size() or 1 in plane_labels_stack.size():
                continue

            self.__sc_metric(intersection, union, plane_labels_stack, gt_plane_labels_stack, mask[batch_id])

            # Calculate depth and angle recall
            self.__eval_depth_recall(intersection[1:, 1:], union[1:, 1:], intersection_mask[:, :, 1:, 1:], gt_plane_labels_stack[:, :, 1:],
                                     depth[batch_id].squeeze(), gt_depth[batch_id].squeeze(),
                                     self.__plane_depth_threshold)
            self.__eval_normal_recall(intersection[1:, 1:], union[1:, 1:], intersection_mask[:, :, 1:, 1:], normals[batch_id],
                                      gt_normals[batch_id], gt_plane_labels_stack[:, :, 1:], self.__plane_normal_threshold)

    def get_all_results(self):
        """Returns all results calculated with this Metrics class instance.

        Returns:
            dict: Dictionary containing result structure.
        """
        return {"plane_metrics": self.get_plane_metrics(), "estimation_metrics": self.get_estimation_metrics()}

    def get_plane_metrics(self):
        """Returns segmentation metrics and plane recall calculated with this Metrics class instance.

        Returns:
            dict: Sub-dictionary of the plane_metrics branch.
        """
        return {"IOU": self.__iou.avg, "RI": self.__ri.avg, "SC": self.__sc.avg, "VOI": self.__voi.avg,
                "perc_found_planes": self.__overlapping_planes.avg, "perc_found_pixels": self.__overlapping_pixels.avg}

    def get_estimation_metrics(self):
        """Returns the depth and angle metrics calculated with this Metrics class instance.

        Returns:
            dict: Dictionary containing the depth and surface normal metrics.
        """
        return {"depth": self.get_depth_metrics(), "normals": self.get_normal_metrics()}

    def get_depth_metrics(self):
        """Returns the depth metrics calculated with this Metrics class instance.

        Returns:
            dict: Dictionary containing the depth estimation results.
        """
        return {"REL": self.__rel.avg, "RMSE": np.sqrt(self.__rms.avg), "LOG10": self.__log10.avg,
                "delta1": self.__delta1.avg, "delta2": self.__delta2.avg, "delta3": self.__delta3.avg}

    def get_normal_metrics(self):
        """Returns the normal metrics calculated with this Metrics class instance.

        Returns:
            dict: Dictionary containing the surface normal estimation results.
        """
        return {"angle_accuracy": self.__angle_accuracy.avg, "RMSE": np.sqrt(self.__angle_rms.avg)}

    def __eval_depth_recall(self, intersection, union, intersection_mask, gt_labels_stack,
                            depth, gt_depth, depth_threshold=(0.5,)):
        """Calculates the plane recall parameter based on a depth threshold.

        Parameters:
            intersection (torch.Tensor): The intersection values between all planes (
            union (torch.Tensor): The union values between all planes (
            intersection_mask (torch.Tensor): The mask containing a stack of all intersections between planes (
            gt_labels_stack (torch.Tensor): A stack of all ground truth planes. Each layer contains only one plane
            depth (torch.Tensor): Depth estimation of the network (1, h, w)
            gt_depth (torch.Tensor): Ground truth depth (1, h, w)
            depth_threshold (list): List of all threshold [m] to be used to calculate the recall. (Default: 0.5)
        """
        depth_diff = torch.abs(depth - gt_depth).unsqueeze(-1).unsqueeze(-1)
        plane_diffs = (depth_diff * intersection_mask).sum(0).sum(0) / torch.clamp(intersection, min=1e-4)
        self.__calc_recall(intersection, union, gt_labels_stack, plane_diffs, depth_threshold)

    def __eval_normal_recall(self, intersection, union, intersection_mask, normals, gt_normals,
                             gt_labels_stack, angle_threshold=(11.25,)):
        """Calculates the plane recall parameter based on an angle threshold.

        Parameters:
            intersection (torch.Tensor): The intersection values between all planes (
            union (torch.Tensor): The union values between all planes (
            intersection_mask (torch.Tensor): The mask containing a stack of all intersections between planes (
            gt_labels_stack (torch.Tensor): A stack of all ground truth planes. Each layer contains only one plane
            normals (torch.Tensor): Surface normal estimation of the network (3, h, w)
            gt_normals (torch.Tensor): Ground truth surface normals (3, h, w)
            angle_threshold (list): List of all threshold [°] to be used to calculate the recall. (Default: 11.25)
        """
        cos = torch.nn.CosineSimilarity(dim=0, eps=0)
        angle_diff = torch.arccos(cos(normals, gt_normals + 1e-3)) / np.pi * 180
        plane_diffs = (angle_diff.unsqueeze(-1).unsqueeze(-1) * intersection_mask).sum(0).sum(0) / torch.clamp(intersection, min=1e-4)
        self.__calc_recall(intersection, union, gt_labels_stack, plane_diffs, angle_threshold)

    def __calc_recall(self, intersection, union, gt_labels_stack, diff_matrix, threshold_list):
        """This functions actually calculates the plane recall.

        Parameters:
            intersection (torch.Tensor): The intersection values between all planes (
            union (torch.Tensor): The union values between all planes (
            gt_labels_stack (torch.Tensor): A stack of all ground truth planes. Each layer contains only one plane
            diff_matrix (torch.Tensor): Matrix incorporating the mean depth or angle error between each plane and gt plane.
            threshold_list (list): List of all threshold to be used to calculate the recall.
        """
        # Only consider planes surpassing the iou threshold
        plane_iou_mask = (self.__planes_iou(intersection, union) > self.__iou_threshold).float()

        # Count the pixels and planes for recall calculation
        plane_areas = gt_labels_stack.sum(0).sum(0)
        gt_number_planes = gt_labels_stack.size(-1)

        for threshold in threshold_list:
            # calculate recall for each threshold in the list
            pixel_recall = torch.min((intersection * (diff_matrix <= threshold).float() * plane_iou_mask).sum(1),
                                     plane_areas).sum() / plane_areas.sum()
            plane_recall = (torch.min((diff_matrix * plane_iou_mask + 1e6 * (1 - plane_iou_mask)), dim=1)[0] <
                            threshold).sum() / gt_number_planes

            # save values
            self.__overlapping_pixels.update({str(threshold): pixel_recall.item()})
            self.__overlapping_planes.update({str(threshold): plane_recall.item()})

    def __iou_metric(self, intersection, union):
        """Calculates the IOU metric.

        Parameters:
            intersection (torch.Tensor): Intersection matrix (
            union (torch.Tensor): Union matrix (
        """
        iou_planes = self.__planes_iou(intersection, union)
        iou = iou_planes.max(dim=1)[0].mean()
        self.__iou.update(iou.item())

    def __planes_iou(self, intersection, union):
        """Calculates the IOU value for each plane combination.

        Parameters:
            intersection (torch.Tensor): Intersection matrix (
            union (torch.Tensor): Union matrix (
        """
        return intersection / torch.clamp(union, min=1e-4)

    def __rand_index_metric(self, labels, gt_labels, mask):
        """Calculates the rand index (RI).

        Parameters:
            labels (torch.Tensor): Predicted plane segmentation (1, h, w)
            gt_labels (torch.Tensor): Ground truth plane segmentation (1, h, w)
            mask (torch.Tensor): nan-mask (1, h, w)
        """
        gt = np.reshape(gt_labels.squeeze().cpu().numpy(), (-1))[np.reshape(mask.cpu().numpy(), (-1))]
        labels = np.reshape(labels.squeeze().cpu().numpy(), (-1))[np.reshape(mask.cpu().numpy(), (-1))]
        ri = metrics.rand_score(gt, labels)
        self.__ri.update(ri)

    def __voi_metric(self, intersection):
        """Calculates the variation of information metric.

        Parameters:
            intersection (torch.Tensor): Intersection matrix (
        """
        N = intersection.sum()
        joint = intersection / N
        marginal_2 = joint.sum(0)
        marginal_1 = joint.sum(1)
        H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
        H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

        B = (marginal_1.unsqueeze(-1) * marginal_2)
        log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float()
        MI = (joint * log2_quotient).sum()
        voi = H_1 + H_2 - 2 * MI

        self.__voi.update(voi.item())

    def __sc_metric(self, intersection, union, labels, gt_labels, mask):
        """Calculates the SC metric.

        Parameters:
            intersection (torch.Tensor): Intersection matrix (
            union (torch.Tensor): Union matrix (
            labels (torch.Tensor): Predicted plane segmentation (1, h, w)
            gt_labels (torch.Tensor): Ground truth plane segmentation (1, h, w)
            mask (torch.Tensor): nan-mask (1, h, w)
        """
        N = intersection.sum()
        iou = self.__planes_iou(intersection, union)
        mask = mask.squeeze(-1)
        sc = ((iou.max(-1)[0] * torch.clamp((gt_labels * mask).sum(0).sum(0), min=1e-4)).sum() / N + (
                    iou.max(0)[0] * torch.clamp((labels * mask).sum(0).sum(0), min=1e-4)).sum() / N) / 2
        self.__sc.update(sc.item())

    def __evaluate_depth_estimation(self, depth, gt_depth, mask):
        """Calculates all the metrics evaluating the depth estimation.

        Parameters:
            depth (torch.Tensor): Estimated depth (b, 1, h, w)
            gt_depth (torch.Tensor): Ground truth depth (b, 1, h, w)
            mask (torch.Tensor): nan-mask (b, 1, h, w)
        """
        nValidElement = torch.sum(mask)
        batch_size = depth.size(0)
        if (nValidElement.data.cpu().numpy() > 0):
            diffMatrix = torch.abs(depth - gt_depth)

            self.__rms.update((torch.pow(diffMatrix, 2).sum() / nValidElement).item(), batch_size)

            realMatrix = torch.div(diffMatrix, gt_depth)
            realMatrix[torch.logical_not(mask)] = 0
            realMatrix[realMatrix.isinf()] = 0
            self.__rel.update((torch.sum(realMatrix) / nValidElement).item(), batch_size)

            LG10Matrix = torch.abs(torch.div(torch.log(depth), np.log(10)) - torch.div(torch.log(gt_depth), np.log(10)))
            LG10Matrix[torch.logical_not(mask)] = 0
            LG10Matrix[LG10Matrix.isnan()] = 0
            LG10Matrix[LG10Matrix.isinf()] = 0
            self.__log10.update((torch.sum(LG10Matrix) / nValidElement).item(), batch_size)

            yOverZ = torch.div(depth, gt_depth)
            zOverY = torch.div(gt_depth, depth)

            maxRatio = self.__maxOfTwo(yOverZ, zOverY)

            self.__delta1.update((torch.sum(torch.le(maxRatio, 1.25).float()) / nValidElement).item(), batch_size)
            self.__delta2.update((torch.sum(torch.le(maxRatio, np.power(1.25, 2)).float()) / nValidElement).item(), batch_size)
            self.__delta3.update((torch.sum(torch.le(maxRatio, np.power(1.25, 3)).float()) / nValidElement).item(), batch_size)

    def __maxOfTwo(self, x, y):
        z = x.clone()
        maskYLarger = torch.lt(x, y)
        z[maskYLarger.detach()] = y[maskYLarger.detach()]
        return z

    def __evaluate_normal_estimation(self, normals, gt_normals, mask):
        """Calculates all the metrics evaluating the surface normal estimation.

        Parameters:
            normals (torch.Tensor): Estimated surface normals (b, 3, h, w)
            gt_normals (torch.Tensor): Ground truth depth (b, 3, h, w)
            mask (torch.Tensor): nan-mask (b, 1, h, w)
        """
        nValidElement = torch.sum(mask)
        batch_size = normals.size(0)
        if (nValidElement.data.cpu().numpy() > 0):
            # calculate the angle error
            cos = torch.nn.CosineSimilarity(dim=1, eps=0)
            angles = torch.arccos(cos(normals, gt_normals + 1e-3)) / np.pi * 180
            angles = angles.unsqueeze(1)
            # count how many are within the following thresholds
            self.__angle_accuracy.update({"11.25": (torch.sum((angles < 11.25)*mask)/nValidElement).item()})
            self.__angle_accuracy.update({"22.5": (torch.sum((angles < 22.5)*mask)/nValidElement).item()})
            self.__angle_accuracy.update({"30": (torch.sum((angles < 30)*mask)/nValidElement).item()})

            # Calculate the RMS of the angles
            self.__angle_rms.update((torch.pow(angles*mask, 2).sum() / nValidElement).item())
