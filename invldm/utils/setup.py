import os
import yaml
import shutil
import argparse
import random
import numpy as np
import torch
import sys
import logging
import multiprocessing

from .utils import dict2namespace, namespcae_summary_ticket
from ..loggers.utils import _instance_logger

sys_logger = logging.getLogger("ray")


def setup_train():
    # getting cl and yml args
    args = parse_args()

    # add default args to missing args
    default_missing_args(args)

    # check devices
    check_devices(args)

    # check overwrite:
    check_overwrite(args)

    # check and adjust dataset files
    check_dataset_file(args)

    # checking tracking tool
    check_tracking_tool(args)

    # create relevant folders
    create_training_experiment_folders(args)

    # setup seeds and devices
    set_seed(args.run.seed)

    # set up loggers
    setup_logger(args)

    # if condition requested but path not passed, look at working directory
    if args.data.condition.mode is not None and args.data.condition.path is None:
        args.data.condition.path = os.getcwd()

    # for training we must make sure the sampling_only flags are off
    args.autoencoder.sampling_only = False
    args.diffusion.sampling_only = False

    # save updated args as config in log folder
    with open(os.path.join(args.run.log_folder, "config.yml"), "w") as f:
        yaml.dump(args, f, default_flow_style=False)

    # Set up tracker -- note this should come AFTER the yaml dump
    setup_model_tracker(args)

    # std out experiment summary ticket
    if args.run.resume_training:
        config = "\n\n ------------------------------ RESUMED CONFIGURATION ------------------------------ " + namespcae_summary_ticket(args)
    else:
        config = "\n\n ---------------------------------- CONFIGURATION ---------------------------------- " + namespcae_summary_ticket(args)
    sys_logger.info(config)

    return args


def setup_sampling():
    # getting cl and yml args
    args = parse_args()

    # add default args to missing args
    default_missing_args(args)

    # check devices
    check_devices(args)

    # create relevant folders
    create_sampling_experiment_folders(args)

    # setup seeds and devices
    set_seed(args.run.seed)

    # set up loggers
    setup_logger(args)

    # for sampling, we must make sure the sampling_only flags are on
    args.autoencoder.sampling.sampling_only = True
    args.diffusion.sampling.sampling_only = True

    # save updated args as config in log folder
    with open(os.path.join(args.run.log_folder, "config.yml"), "w") as f:
        yaml.dump(args, f, default_flow_style=False)

    # Set up tracker -- note this should come AFTER the yaml dump
    setup_model_tracker(args)

    # std out experiment summary ticket
    config = "\n\n ---------------------------------- SAMPLING CONFIGURATION ---------------------------------- " + namespcae_summary_ticket(args)
    sys_logger.info(config)

    return args


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any
    randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return True


def parse_args():
    # parse command line arguments
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file for training or sampling"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="exps",
        help="Path for saving running related data."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of experiment and also the log folder.",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="A string for experiment comment"
    )
    parser.add_argument(
        "--jobid",
        type=int,
        default=-1,
        help="Store job ID for hpc systems"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to which run. Options: cuda or cpu. If cuda, specify\
            gpu_ids with flag"
    )
    parser.add_argument(
        "--gpu_ids",
        type=list,
        default=None,
        help="List of gpu ids to be used in DataParallel. If None, all will be\
            used"
    )
    parser.add_argument(
        "--ray_num_workers",
        type=int,
        default=4,
        help="Number of process workers used for ray. If `device` is set \
            to `cpu` this number refers to number of processors. Conversely, \
            if `device` is set to `cuda` this number refers to number of gpus. \
            If `device` is set to `cuda` and `gpu_ids` is passed, this value \
            is overridden by the length of `gpu_ids` list. \
            If None, max number of processors or gpus will be used."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Whether to resume training. Used in conjunction\
            with checkpoint flag in yaml config. If checkpoint not specified, will use latest model."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to \
            overwrite experiment of same name"
    )
    parser.add_argument(
        "-y", action="store_true", help="Automatic 'yes' to user input\
            questions"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )

    run_args = parser.parse_args()
    run_args.exp_folder = os.path.abspath(os.path.join(run_args.logdir,
                                       run_args.name))
    run_args.log_folder = os.path.abspath(os.path.join(run_args.exp_folder,
                                       "logs"))

    # read config, add run args to namespace
    args = parse_config(run_args.config)
    args.run = run_args

    return args


def parse_config(config_file):
    try:
        with open(config_file, "r") as f:
            config = dict2namespace(yaml.safe_load(f))
    except yaml.constructor.ConstructorError:
        sys_logger.warn(f"Loading {config_file} unsafely with YML.")
        with open(config_file, "r") as f:
            config = yaml.unsafe_load(f)
        try:
            config = dict2namespace(config)
        except AttributeError:
            pass
    
    return config


def check_dataset_file(args):
    # Check dataset_path:
    try:
        dataset_path = args.data.dataset_path
        if dataset_path is None or not dataset_path:
            raise AttributeError
    except AttributeError:
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    '..', "datasets"))
        args.data.dataset_path = dataset_path

    # Check dataset file
    try:
        open(os.path.join(args.data.dataset_path,
                          args.data.dataset.lower() + "_dataset.py"), "r")
    except AttributeError:
        raise Exception(AttributeError, "Dataset name must be passed in config\
                        file")
    except FileNotFoundError:
        msg = f"File {args.data.dataset.lower()}_dataset.py must exist in\
                            {args.data.dataset_path}"
        sys_logger.exception(msg)
        raise Exception(FileNotFoundError, msg)
    return args


def check_devices(args):
    if args.run.device == "cpu" or "cuda" in args.run.device:
        pass
    else:
        new_device = "cuda" if torch.cuda.is_available() else "cpu"
        sys_logger.warn(f"Device '{args.run.device}' can't be recognised. Changing devide to {new_device}.")
        args.run.device = new_device

    gpu_ids = []
    if args.run.gpu_ids and "cuda" in args.run.device:
        for id in args.run.gpu_ids:
            try:
                gpu_ids.append(int(id))
            except ValueError:
                pass        

        # Make only these gpus visible, adjust 
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids)) 
        args.run.ray_num_workers = len(gpu_ids)
    args.run.gpu_ids = gpu_ids

    # Adjust ray_num_workers
    if args.run.ray_num_workers is None:
        args.run.ray_num_workers = multiprocessing.cpu_count() if args.run.device == "cpu" else torch.cuda.device_count()

    return args


def check_tracking_tool(args):
    if args.logging.tool:
        track_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '..', "loggers", args.logging.tool.lower() + "_logger.py"))
        try:
            open(track_file, "r")
        except FileNotFoundError:
            raise Exception(FileNotFoundError,
                            f"Expecting file '{track_file}' to exist, but not found")
    return args


def default_missing_args(args):
    default_args = parse_config(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "default_config.yml"))
    
    def default(args, default_args):
        for key in default_args.__dict__:
            default_value = default_args.__dict__[key]
            try:
                args.__dict__[key]
            except KeyError:
                setattr(args, key, default_value)

            if isinstance(default_value, argparse.Namespace):
                _ = default(args.__dict__[key],
                            default_args.__dict__[key])
        return args
    
    args = default(args, default_args)

    args.autoencoder.name = "autoencoder"
    args.diffusion.name = "diffusion"

    return args


def check_overwrite(args):
    if args.run.overwrite:
        if args.run.resume_training:
            args.run.overwrite = False
            sys_logger.warn("Both --resume_training and --overwrite flags passed, turning off overwrite")
        elif os.path.exists(args.run.exp_folder):
            if args.run.y:
                user_input = "y"
            else:
                user_input = input(f"The current directory '{args.run.exp_folder}' is about to be rewritten. Do you wish to proceed (YES/NO) : ")
            
            if user_input.lower() in ['y', 'yes']:
                pass
            elif user_input.lower() in ['n', 'no']:
                sys_logger.info("No overwriting, aborting script.")
                sys.exit()
            else:
                check_overwrite(args)
    return args


def create_folder(folder, overwrite=False, exception=True):
    try:
        os.makedirs(folder)
    except FileExistsError as e:
        if overwrite:
            sys_logger.info(f"Overwriting {folder} ...")
            shutil.rmtree(folder)
            os.makedirs(folder)
            sys_logger.info(f"Done!")
        else:
            if exception:
                raise Exception(FileExistsError, f"Experiment folder '{folder}' already exists. Use the --overwrite flag if you wish to overwrite or --resume_training flag to continue training from a checkpoint.")
            else:
                pass
    return None


def create_training_experiment_folders(args):
    # Whether to raise overwriting exception when folder already exists
    if args.run.resume_training:
        exception = False
    else:
        exception = True

    # Create experiment folder
    create_folder(args.run.exp_folder, args.run.overwrite, exception)

    # Create logging folder
    create_folder(args.run.log_folder, args.run.overwrite, exception)

    # Create logging folders for autoencoder and diffusion
    autoencoder_log_path = os.path.join(args.run.log_folder, "autoencoder")
    args.autoencoder.log_path = autoencoder_log_path
    create_folder(autoencoder_log_path, args.run.overwrite, exception)

    diffusion_log_path = os.path.join(args.run.log_folder, "diffusion")
    args.diffusion.log_path = diffusion_log_path
    create_folder(diffusion_log_path, args.run.overwrite, exception)

    # Create and checkpoint folder for autoencoder and diffusion
    autoencoder_ckpt_path = os.path.join(autoencoder_log_path, "checkpoints")
    args.autoencoder.ckpt_path = autoencoder_ckpt_path
    create_folder(autoencoder_ckpt_path, args.run.overwrite, exception)

    diffusion_ckpt_path = os.path.join(diffusion_log_path, "checkpoints")
    args.diffusion.ckpt_path = diffusion_ckpt_path
    create_folder(diffusion_ckpt_path, args.run.overwrite, exception)

    # Create model tracking management folder
    if args.logging.tool:
        track_folder = os.path.join(args.run.exp_folder, args.logging.tool)
        args.logging.track_folder = track_folder
        create_folder(track_folder, args.run.overwrite, exception)

    # Create samples folders
    if args.diffusion.training.sampling_freq > 0:
        samples_folder = os.path.join(diffusion_log_path, "samples")
        args.diffusion.samples_path = samples_folder
        create_folder(samples_folder, args.run.overwrite, exception)
    if args.autoencoder.training.sampling_freq > 0:
        samples_folder = os.path.join(autoencoder_log_path, "samples")
        args.autoencoder.samples_path = samples_folder
        create_folder(samples_folder, args.run.overwrite, exception)

    # Create recon folders
    if (args.autoencoder.training.save_recon_freq > 0 or args.autoencoder.validation.save_recon_freq > 0):
        recon_folder = os.path.join(autoencoder_log_path, "recons")
        args.autoencoder.recon_path = recon_folder
        create_folder(recon_folder, args.run.overwrite, exception)

    return None


def create_sampling_experiment_folders(args):

    # Check experiment exists
    assert os.path.exists(args.run.exp_folder), f"Experiment folder '{args.run.exp_folder}' not found. "
    assert os.path.exists(args.run.log_folder), f"Experiment logging folder '{args.run.log_folder}' not found. "

    autoencoder_log_path = os.path.join(args.run.log_folder, "autoencoder")
    args.autoencoder.log_path = autoencoder_log_path
    assert os.path.exists(autoencoder_log_path), f" Autoencoder logging folder '{autoencoder_log_path}' not found. " 

    diffusion_log_path = os.path.join(args.run.log_folder, "diffusion")
    args.diffusion.log_path = diffusion_log_path
    assert os.path.exists(diffusion_log_path), f" Diffusion logging folder '{diffusion_log_path}' not found. " 

    autoencoder_ckpt_path = os.path.join(autoencoder_log_path, "checkpoints")
    args.autoencoder.ckpt_path = autoencoder_ckpt_path
    assert os.path.exists(autoencoder_ckpt_path), f" Autoencoder checkpoint folder '{autoencoder_ckpt_path}' not found. " 

    diffusion_ckpt_path = os.path.join(diffusion_log_path, "checkpoints")
    args.diffusion.ckpt_path = diffusion_ckpt_path
    assert os.path.exists(diffusion_ckpt_path), f" Autoencoder checkpoint folder '{diffusion_ckpt_path}' not found. " 

    # Create samples folders
    samples_folder = os.path.join(args.run.exp_folder, "samples")
    args.run.samples_folder = samples_folder
    create_folder(samples_folder, args.run.overwrite, exception=False)

    return None


def setup_logger(args):
    level = getattr(logging, args.run.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("Logger level {} not supported".format(args.run.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.run.log_folder, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    sys_logger.addHandler(handler1)
    sys_logger.addHandler(handler2)
    sys_logger.setLevel(level)

    args.logging.logger = sys_logger
    args.run.pid = os.getpid()

    sys_logger.info("Writing log file to {}".format(args.run.log_folder))
    sys_logger.info("Exp instance id = {}".format(args.run.pid))
    sys_logger.info("Exp comment = {}".format(args.run.comment))

    return args


def setup_model_tracker(args):
    if args.logging.tool:
        args.logging.tracker = _instance_logger(args)
    else:
        args.logging.tracker = None
    return args
