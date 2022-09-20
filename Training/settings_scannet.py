from Utilities.DatasetEnum import DatasetEnum


class TrainingSettings:
    # Training
    epochs = 21
    start_epoch = 0
    lr = 0.001
    momentum = 0.85
    weight_decay = 1e-4
    dataset_dir = ""
    network_file = None
    lr_adjustment = 7  # after n epochs 1/10th
    batch_size_1gpu = 3
    batch_size_2gpu = 10
    batch_size_4gpu = 20
    batch_size_8gpu = 40
    training_datasets = DatasetEnum.SCANNET  # NyuV2 is not supported
    max_training_depth = 10

    # Tensorboard Logging
    tb_folder_name = "runs"
    tb_run_name = "pec_scannet"
    tb_log_freq = 50
    eval_batch_size = 10

    # online evaluation
    evaluate_each_epoch = False
    # these only apply if true:
    tb_val_folder_name = "runs_embedding_val"


class EvaluationSettings:
    evaluate_hypersim = False
    evaluate_scannet = True
    dataset_dir = TrainingSettings.dataset_dir
    network_file = "../PEC-SCANNet.tar"
    n_example_pictures = 15
    example_pictures_random_seed = 222
    batch_size = 10
    tensorboard_log_dir = "network_eval"
    tensorboard_eval_name = "pec_scannet"


class SystemSettings:
    # System settings
    num_cpus = 8
