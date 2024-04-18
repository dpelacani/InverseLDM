import torch
import logging
import ray.train.torch

from .autoencoder_runner import AutoencoderRunner
from .diffusion_runner import DiffusionRunner

from ..seismic.utils import _instance_conditioner
from ..datasets.utils import (_wrap_tensor_dataset, _instance_dataloader)


class Sampler():
    def __init__(self, args):
        self.args = args

        # Datasets
        self.autoencoder_sampling_dataset = _wrap_tensor_dataset(
            torch.randn((self.args.autoencoder.sampling.n_samples,
                                    self.args.autoencoder.sampling.input_channels,
                                    self.args.autoencoder.sampling.input_image_size,
                                    self.args.autoencoder.sampling.input_image_size))
        )
        self.diffusion_sampling_dataset = _wrap_tensor_dataset(
            torch.randn((self.args.diffusion.sampling.n_samples,
                                    self.args.autoencoder.sampling.input_channels,
                                    self.args.autoencoder.sampling.input_image_size,
                                    self.args.autoencoder.sampling.input_image_size))
        )

        # Dataloaders
        self.autoencoder_sample_dataloader = _instance_dataloader(
            self.args.autoencoder.sampling, self.autoencoder_sampling_dataset
        )
        self.autoencoder_sample_dataloader = ray.train.torch.prepare_data_loader(self.autoencoder_sample_dataloader)

        self.diffusion_sample_dataloader = _instance_dataloader(
            self.args.diffusion.sampling, self.diffusion_sampling_dataset
        )
        self.diffusion_sample_dataloader = ray.train.torch.prepare_data_loader(self.diffusion_sample_dataloader)

        # Autoencoder runner, load pre-trained, eval mode
        assert args.autoencoder.sampling.sampling_only
        self.autoencoder = AutoencoderRunner(
            args=args.autoencoder,
            args_run=args.run,
            args_logging=args.logging,
            sample_loader=self.autoencoder_sample_dataloader
        )
        self.autoencoder.load_checkpoint(
            self.autoencoder.checkpoint_path(),
            model_only=True
        )
        self.autoencoder.model.module.model.eval()

        # Diffusion runner, load pre-trained, eval mode
        assert args.diffusion.sampling.sampling_only
        self.diffusion = DiffusionRunner(
            autoencoder=self.autoencoder.model.module.model,
            args=args.diffusion,
            args_run=args.run,
            args_logging=args.logging,
            sample_loader=self.diffusion_sample_dataloader
        )
        self.diffusion.load_checkpoint(
            self.diffusion.checkpoint_path(),
            model_only=True
        )
        self.diffusion.model.module.ldm.eval()

    def sample(self):
        logging.info(" ---- Autoencoder Sampling ---- ")
        self.autoencoder.sample()
        logging.info(" ---- Diffusion Sampling ---- ")
        self.diffusion.sample()
        logging.info(" ---- Sampling Concluded without Errors ---- ")

