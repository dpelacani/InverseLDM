import os

import torch
import logging
import ray
import trainy

from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

from torchsummary import summary
from datetime import datetime

from .autoencoder_runner import AutoencoderRunner
from .diffusion_runner import DiffusionRunner

from ..seismic.utils import _instance_conditioner
from ..datasets.utils import (_instance_dataset, _instance_dataloader,
                            _split_valid_dataset)

sys_logger = logging.getLogger("ray")


class Trainer():
    def __init__(self, args):
        self.args = args

        # Ray scaling config
        self.ray_scaling_config = ScalingConfig(num_workers=args.run.ray_num_workers, use_gpu="cuda" in args.run.device)
        self.ray_running_config = RunConfig(name="ray", storage_path=args.run.log_folder)

        # Datasets
        self.dataset = _instance_dataset(self.args.data)
        self.autoencoder_train_dataset, self.autoencoder_valid_dataset = \
            _split_valid_dataset(args.autoencoder, self.dataset)
        self.diffusion_train_dataset, self.diffusion_valid_dataset = \
            _split_valid_dataset(args.diffusion, self.dataset)

        # Dataloaders
        self.autoencoder_train_dataloader = _instance_dataloader(
            self.args.autoencoder.training, self.autoencoder_train_dataset
        )
        self.autoencoder_valid_dataloader = _instance_dataloader(
            self.args.autoencoder.validation, self.autoencoder_valid_dataset
        )
        self.diffusion_train_dataloader = _instance_dataloader(
            self.args.diffusion.training, self.diffusion_train_dataset
        )
        self.diffusion_valid_dataloader = _instance_dataloader(
            self.args.diffusion.validation, self.diffusion_valid_dataset
        )
        # Model trainers
        self.autoencoder = AutoencoderRunner(
            args=args.autoencoder,
            args_run=args.run,
            args_logging=args.logging,
            train_loader=self.autoencoder_train_dataloader,
            valid_loader=self.autoencoder_valid_dataloader,
        )

        self.diffusion = DiffusionRunner(
            autoencoder=self.autoencoder.model.model,
            args=args.diffusion,
            args_run=args.run,
            args_logging=args.logging,
            train_loader=self.diffusion_train_dataloader,
            valid_loader=self.diffusion_valid_dataloader,
        )

    def ray_train_autoencoder(self):
        # Ray dataloader wrappers
        self.autoencoder.train_loader = ray.train.torch.prepare_data_loader(self.autoencoder_train_dataloader)
        self.autoencoder.valid_loader = ray.train.torch.prepare_data_loader(self.autoencoder_valid_dataloader)

        # Ray model wrappers --> Instances of nn.DistributedDataParallel
        ddp_args=dict(find_unused_parameters=False)
        self.autoencoder.model = ray.train.torch.prepare_model(self.autoencoder.model,
                                                               parallel_strategy_kwargs=ddp_args)
        if "d_model" in self.autoencoder.__dict__.keys():
            self.autoencoder.d_model = ray.train.torch.prepare_model(self.autoencoder.d_model,
                                                                     parallel_strategy_kwargs=ddp_args)

        # Call training
        self.autoencoder.train()

    
    def ray_train_diffusion(self):
        # Ray dataloader wrappers
        self.diffusion.train_loader = ray.train.torch.prepare_data_loader(self.diffusion_train_dataloader)
        self.diffusion.valid_loader = ray.train.torch.prepare_data_loader(self.diffusion_valid_dataloader)

        # Ray model wrappers --> Instances of nn.DistributedDataParallel
        ddp_args=dict(find_unused_parameters=True)
        self.diffusion.model.autoencoder = ray.train.torch.prepare_model(self.diffusion.model.autoencoder,
                                                                         parallel_strategy_kwargs=ddp_args)
        self.diffusion.model.ldm.autoencoder = self.diffusion.model.autoencoder
        self.diffusion.model = ray.train.torch.prepare_model(self.diffusion.model,
                                                             parallel_strategy_kwargs=ddp_args)

        # Call training
        self.diffusion.train()

    def train(self):
        s_time = datetime.now()

        # sys_logger.info(" ---- Dataset ---- ")
        # sys_logger.info(self.dataset)

        # logging.info(" ---- Model - Autoencoder ----")
        # sample = self.dataset[0]
        # if isinstance(sample, tuple):
        #     sample = sample[0]
        # sample = sample.to(self.autoencoder.device)
        # logging.info(summary(model=self.autoencoder.model, input_data=sample.shape, device=self.autoencoder.device))

        # logging.info(" ---- Model - Diffusion ----")
        # if self.args.diffusion.training.n_epochs > 0:
        #     with torch.no_grad():
        #         _ = self.diffusion.model.autoencoder.encode(sample.unsqueeze(0).float())
        #         embbeded_sample = self.diffusion.model.autoencoder.sample().squeeze(0).to(self.diffusion.device)
        #     logging.info(summary(model=self.diffusion.model, input_data=embbeded_sample.shape, device=self.diffusion.device))
        
        sys_logger.info(" ---- Autoencoder Training ---- ")
        # ray.init()
        ray_autoencoder_trainer = TorchTrainer(self.ray_train_autoencoder,
                                               scaling_config=self.ray_scaling_config,
                                               run_config=self.ray_running_config)
        ray_autoencoder_trainer.fit()

        if self.args.diffusion.training.n_epochs > 0:
            # ray.init()
            sys_logger.info(" ---- Diffusion Training ---- ")
            ray_diffusion_trainer = TorchTrainer(self.ray_train_diffusion,
                                                 scaling_config=self.ray_scaling_config,
                                                 run_config=self.ray_running_config)
            ray_diffusion_trainer.fit()
            # ray.stop()

        t_time = datetime.now() - s_time
        h, m, s = str(t_time).split(".")[0].split(":")
        sys_logger.info(f" ---- Training Concluded in {h}h {m}m {s}s without Errors ---- ")
