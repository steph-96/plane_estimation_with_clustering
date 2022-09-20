import argparse
from os import path, makedirs

import tensorflow as tf
import numpy as np
from Utilities.util import Utility

parser = argparse.ArgumentParser(description='Transforms Scannnet tfrecords file to single hdf5 files')
parser.add_argument("--dataset_dir", help='path to the dataset', required=True)
parser.add_argument("--file", help='path to the download directory', required=True)
args = parser.parse_args()

HEIGHT = 192
WIDTH = 256

def map_fct(data):
    return tf.io.parse_single_example(
        data,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'image_path': tf.io.FixedLenFeature([], tf.string),
            'num_planes': tf.io.FixedLenFeature([], tf.int64),
            'plane': tf.io.FixedLenFeature([20 * 3], tf.float32),
            'segmentation_raw': tf.io.FixedLenFeature([], tf.string),
            'depth': tf.io.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
            'normal': tf.io.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
            'semantics_raw': tf.io.FixedLenFeature([], tf.string),
            'boundary_raw': tf.io.FixedLenFeature([], tf.string),
            'info': tf.io.FixedLenFeature([4 * 4 + 4], tf.float32),
        })


data = tf.data.TFRecordDataset(args.file).map(map_fct).as_numpy_iterator()

if not path.exists(args.dataset_dir):
    makedirs(args.dataset_dir)

for i, next_data in enumerate(data):
    number = f"{i}".zfill(6)
    depth = np.array(next_data['depth']).reshape((HEIGHT, WIDTH))
    normal = np.array(next_data['normal']).reshape((HEIGHT, WIDTH, 3))
    image = np.array(tf.io.decode_raw(next_data['image_raw'], tf.uint8)).reshape((HEIGHT, WIDTH, 3)) / 255
    segmentation = np.array(tf.io.decode_raw(next_data['segmentation_raw'], tf.uint8)).reshape(
        (HEIGHT, WIDTH))
    plane_para = next_data['plane'].reshape((20, 3))

    plane_d = np.linalg.norm(plane_para, axis=1)
    normals = plane_para / np.expand_dims(plane_d, 1)
    normals[np.isnan(normals)] = 0
    segment_list = list()
    normal_list = list()
    for segment in range(0, 20):
        segment_list.append((segmentation == segment) * plane_d[segment])
        normal_list.append(np.expand_dims(segmentation == segment, 2) * normals[segment])

    normal = np.stack(normal_list).sum(0)
    plane_d = np.stack(segment_list).sum(0)
    segmentation += 1
    segmentation[segmentation == 21] = 0
    planes = np.moveaxis(np.stack((plane_d, segmentation)), 0, -1)

    Utility.save_hdf5_file(depth, path.join(args.dataset_dir, f"depth_{number}.hdf5"))
    Utility.save_hdf5_file(normal, path.join(args.dataset_dir, f"normal_{number}.hdf5"))
    Utility.save_hdf5_file(image, path.join(args.dataset_dir, f"image_{number}.hdf5"))
    Utility.save_hdf5_file(planes, path.join(args.dataset_dir, f"planes_{number}.hdf5"))


