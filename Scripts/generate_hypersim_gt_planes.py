"""Short, simple script to generate ground truth plane segmentation for Hypersim."""

import argparse
import multiprocessing
import tqdm

from os import path

from Training.Dataloader.HypersimDataset import HypersimDataset

parser = argparse.ArgumentParser(description='Generate planes parameters and labels ground truth')
parser.add_argument("--dataset_dir", help='path to the dataset')
parser.add_argument("--cpu_cores", required=True, type=int)
parser.add_argument("--hypersim", action="store_true")
args = parser.parse_args()

cpu_cores = args.cpu_cores
print(f"cpu cores: {cpu_cores}")

global meanshift_cores

if args.hypersim:
    meanshift_cores = 1

    if args.dataset_dir is None:
        dataset_dir = path.join(path.join(path.dirname(path.abspath(__file__)), ".."), "Dataset")
    else:
        dataset_dir = args.dataset_dir

    dataset_train = HypersimDataset(dataset_dir, "train")
    dataset_test = HypersimDataset(dataset_dir, "val")


    def run_gt_generator_train(id):
        dataset_train.generate_gt(id, meanshift_cores)


    def run_gt_generator_test(id):
        dataset_test.generate_gt(id, meanshift_cores)

    print("Starting with hypersim training data")

    with multiprocessing.Pool(processes=int(cpu_cores/meanshift_cores)) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(run_gt_generator_train, range(len(dataset_train))),
                           total=len(dataset_train)):
            pass

    print("Done with hypersim training data")

    print("Starting with hypersim testing data")

    with multiprocessing.Pool(processes=int(cpu_cores/meanshift_cores)) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(run_gt_generator_test, range(len(dataset_test))),
                           total=len(dataset_test)):
            pass

    print("Done with hypersim testing data")
