"""This file contains the evaluation script for a trained network."""

import numpy as np
import pyransac
import torch

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Evaluation.metrics import Metrics
from Training.Dataloader import loaddata
from Training.models import Model
from Training.settings import EvaluationSettings
from Utilities.DatasetEnum import DatasetEnum
from Utilities.util import Utility
from Utilities.Visualize import Visualisation as vs


import warnings
warnings.filterwarnings("ignore")


def main():
    # loading SENet model
    model = Model.Model(num_features=2048, block_channel=[256, 512, 1024, 2048])
    model = torch.nn.DataParallel(model).cuda()

    # Setting up the tensorboard logger
    date_time = datetime.now().strftime("%m%d%H%M%S")
    name = date_time+"_"+EvaluationSettings.tensorboard_eval_name
    writer = SummaryWriter(log_dir=EvaluationSettings.tensorboard_log_dir+"/"+name)
    writer.add_text("Filename", EvaluationSettings.network_file)

    # Loading the trained weights
    state_dict = torch.load(EvaluationSettings.network_file)['state_dict']
    model.load_state_dict(state_dict)

    batch_size = EvaluationSettings.batch_size

    if EvaluationSettings.evaluate_hypersim:
        # Hypersim
        dataset = DatasetEnum.HYPERSIM
        # loading data
        test_loader = loaddata.get_testing_data_hypersim(EvaluationSettings.dataset_dir, batch_size)
        # Evaluation
        result_metric = evaluate(test_loader, model, writer, dataset)
        # Logging results to Tensorboard
        log_planes_metric(result_metric.get_plane_metrics(), writer, dataset)
        log_depth_metric(result_metric.get_depth_metrics(), writer, dataset)
        log_normal_metric(result_metric.get_normal_metrics(), writer, dataset)
        print(result_metric.get_all_results())

    if EvaluationSettings.evaluate_scannet:
        # ScanNET
        dataset = DatasetEnum.SCANNET
        # loading data
        test_loader = loaddata.get_testing_data_scannet(EvaluationSettings.dataset_dir, batch_size)
        # Evaluation
        result_metric = evaluate(test_loader, model, writer, dataset)
        # Logging results to Tensorboard
        log_planes_metric(result_metric.get_plane_metrics(), writer, dataset)
        log_depth_metric(result_metric.get_depth_metrics(), writer, dataset)
        log_normal_metric(result_metric.get_normal_metrics(), writer, dataset)
        print(result_metric.get_all_results())


def log_planes_metric(result_dict, writer, dataset, epoch=0):
    """This functions logs the plane metrics in tensorboard.

    Parameters:
        result_dict (dict): The plane metrics dictionary which is logged.
        writer (SummaryWriter): Tensorboard writer instance
        dataset (DatasetEnum): To which dataset the result belongs
    """
    for key, value in result_dict.items():
        if type(value) == dict:
            for subkey, subvalue in value.items():
                if float(subkey) > 1:
                    writer.add_scalar(f"{dataset.name} - Plane Estimation Metric - {key} normals", subvalue, int(100*float(subkey)))
                else:
                    writer.add_scalar(f"{dataset.name} - Plane Estimation Metric - {key} depth", subvalue, int(100*float(subkey)))
        else:
            writer.add_scalar(f"{dataset.name} - Plane Estimation Metric - {key}", value, epoch)


def log_depth_metric(result_dict, writer, dataset, epoch=0):
    """This functions logs the depth metrics in tensorboard.

    Parameters:
        result_dict (dict): The depth metrics dictionary which is logged.
        writer (SummaryWriter): Tensorboard writer instance
        dataset (DatasetEnum): To which dataset the result belongs
    """
    for key, value in result_dict.items():
        writer.add_scalar(f"{dataset.name} - Depth Estimation Metric - {key}", value, epoch)


def log_normal_metric(result_dict, writer, dataset, epoch=0):
    """This functions logs the normal metrics in tensorboard.

    Parameters:
        result_dict (dict): The normal metrics dictionary which is logged.
        writer (SummaryWriter): Tensorboard writer instance
        dataset (DatasetEnum): To which dataset the result belongs
    """
    for key, value in result_dict.get("angle_accuracy").items():
        writer.add_scalar(f"{dataset.name} - Normal Estimation Metric - {key}", value, epoch)
    writer.add_scalar(f"{dataset.name} - Normal Estimation RMSE", result_dict.get("RMSE"), epoch)


@torch.no_grad()
def evaluate(dataloader, model, writer, dataset):
    """Evaluation function.

    This function goes through all the evaluation images and calculates all the evaluation metrics from the metrics
    class

    Parameters:
        dataloader (torch.utils.data.DataLoader): Dataloader with the evaluation images
        model (torch.nn.Module): Trained model to be evaluated
        writer (SummaryWriter): Tensorboard writer instance
        dataset (DatasetEnum): To which dataset the result belongs
    """
    print(f"Evaluating {dataset.name}")
    model.eval()
    metrics = Metrics()

    # Random example images get logged into tensorboard
    n_images = EvaluationSettings.n_example_pictures
    np.random.seed(EvaluationSettings.example_pictures_random_seed)
    random_samples = (np.random.choice(len(dataloader)-1, n_images), np.random.choice(dataloader.batch_size, n_images))
    n = 0

    # Ransac parameters for plane parameter aggregation
    ransac_params = pyransac.RansacParams(samples=3, iterations=2, confidence=0.98, threshold=0.5)

    # Evaluation loop
    for i, sample in enumerate(dataloader):
        image, gt_depth, gt_normals = sample['image'], sample['depth'], sample['normal']
        mask, plane_information = sample['mask'], sample['planes']

        gt_depth = gt_depth.cuda(non_blocking=True)
        image = image.cuda()
        gt_normals = gt_normals.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        plane_information = plane_information.cuda(non_blocking=True)

        gt_plane_para = torch.cat((gt_normals, plane_information[:, 0:1]), dim=1)
        gt_labels = plane_information[:, 1]
        batch_size, _, width, height = gt_depth.size()

        output_plane_para, output_embedding = model(image)

        output_normal = output_plane_para/torch.norm(output_plane_para, dim=1, keepdim=True)

        # Plane segmentation clustering
        output_segmentation = Utility.embedding_segmentation(output_embedding)

        # Ransac plane aggregation
        normals = Utility.aggregate_parameters_ransac(output_segmentation, output_normal, ransac_params)

        # Extracting depth from the estimated surface normal vector
        depth = torch.norm(output_plane_para, dim=1, keepdim=True)
        depth = torch.clamp(depth, 0, 100)

        # Evaluate the output of the network
        metrics.evaluate_output(output_segmentation, gt_labels, depth, gt_depth, normals, gt_normals, mask)

        # Save random images to tensorboard
        if i in random_samples[0]:
            batch_ids = random_samples[1][random_samples[0] == i]
            for idx in batch_ids:
                _write_sample_images(output_segmentation[idx], gt_labels[idx], depth[idx], gt_depth[idx],
                                     image[idx], writer, n, dataset)
                n += 1

    return metrics


def _write_sample_images(labels, gt_labels, depth, gt_depth, image, writer, n, dataset):
    """This function saves the original image, depth comparison and segmentation comparison into tensorboard.

    Parameters:
        labels (torch.Tensor): Outputted plane segmentation
        gt_labels (torch.Tensor): Ground truth plane segmentation
        depth (torch.Tensor): Outputted depth estimation
        gt_depth (torch.Tensor): Ground truth depth
        image (torch.Tensor): Original image the network used to generate output
        writer (SummaryWriter): Tensorboard writer instance
        n (int): the nth image that is saved to tensorboard
        dataset (DatasetEnum): To which dataset the result belongs
    """
    # Saving original image to tensorboard
    image = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=0.5).squeeze().movedim(0, -1).cpu()

    # Prepare the depth estimation comparison by making the same depth the same color
    depth_image, gt_depth_image = vs.prepare_depth_comparison(depth, gt_depth)
    depth_comparison = torch.stack((depth_image, gt_depth_image, image), 0)
    writer.add_images(f"{dataset.name} - Inferred Depth", depth_comparison, n, dataformats='NHWC')

    # Preparing the plane segmentation comparison. The same planes must have the same label.
    plane_label_image, gt_plane_label_image = vs.prepare_label_comparison(labels, gt_labels.unsqueeze(0))
    plane_label_comparison = torch.stack((plane_label_image, gt_plane_label_image, image), 0)
    writer.add_images(f"{dataset.name} - Plane Labels", plane_label_comparison, n, dataformats='NHWC')


if __name__ == "__main__":
    main()
