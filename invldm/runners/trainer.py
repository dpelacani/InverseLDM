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
        with torch.autocast('cuda' if 'cuda' in str(self.args.run.device) else 'cpu'):
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


    def train(self):
        logging.info(" ---- Dataset ---- ")
        logging.info(self.dataset)

        # logging.info(" ---- Model - Autoencoder ----")
        # sample = self.dataset[0]
        # if isinstance(sample, tuple):
        #     sample = sample[0]
        # with torch.no_grad():
        #     sample = sample.to(self.autoencoder.device)
        #     logging.info(summary(model=self.autoencoder.model.module, input_data=sample.shape, device=self.autoencoder.device))

        # logging.info(" ---- Model - Diffusion ----")
        # if self.args.diffusion.training.n_epochs > 0:
        #     with torch.no_grad():
        #         _ = self.diffusion.model.module.autoencoder.encode(sample.unsqueeze(0).float())
        #         embbeded_sample = self.diffusion.model.module.autoencoder.sample().squeeze(0).to(self.diffusion.device)
        #     logging.info(summary(model=self.diffusion.model.module, input_data=embbeded_sample.shape, device=self.diffusion.device))
        
        logging.info(" ---- Autoencoder Training ---- ")
        self.autoencoder.train()

        if self.args.diffusion.training.n_epochs > 0:
            logging.info(" ---- Diffusion Training ---- ")
            self.diffusion.train()

        logging.info(" ---- Training Concluded without Errors ---- ")
