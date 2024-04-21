import torch
import logging
from torchsummary import summary

from .autoencoder_runner import AutoencoderRunner
from .diffusion_runner import DiffusionRunner

from ..seismic.utils import _instance_conditioner
from ..datasets.utils import (_instance_dataset, _instance_dataloader,
                            _split_valid_dataset)


class Trainer():
    def __init__(self, args):
        self.args = args

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
            autoencoder=self.autoencoder.model.module.model,
            args=args.diffusion,
            args_run=args.run,
            args_logging=args.logging,
            train_loader=self.diffusion_train_dataloader,
            valid_loader=self.diffusion_valid_dataloader,
        )

        self.sys_logger = args.logging.logger

    def train(self):
        self.sys_logger.info(" ---- Dataset ---- ")
        self.sys_logger.info(self.dataset)

        self.sys_logger.info(" ---- Model - Autoencoder ----")
        sample = self.dataset[0]
        if isinstance(sample, tuple):
            sample = sample[0]
        sample = sample.to(self.autoencoder.device)
        self.sys_logger.info(summary(model=self.autoencoder.model, input_data=sample.shape, device=self.autoencoder.device))

        self.sys_logger.info(" ---- Model - Diffusion ----")
        if self.args.diffusion.training.n_epochs > 0:
            with torch.no_grad():
                _ = self.diffusion.model.module.autoencoder.encode(sample.unsqueeze(0).float())
                embbeded_sample = self.diffusion.model.module.autoencoder.sample().squeeze(0).to(self.diffusion.device)
            self.sys_logger.info(summary(model=self.diffusion.model, input_data=embbeded_sample.shape, device=self.diffusion.device))
        
        self.sys_logger.info(" ---- Autoencoder Training ---- ")
        self.autoencoder.train()

        if self.args.diffusion.training.n_epochs > 0:
            self.sys_logger.info(" ---- Diffusion Training ---- ")
            self.diffusion.train()

        self.sys_logger.info(" ---- Training Concluded without Errors ---- ")
