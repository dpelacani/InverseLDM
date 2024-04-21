import torch
import torch.nn as nn

from . import BaseRunner

from ..models.utils import (_instance_diffusion_model, _instance_optimiser,
                          _instance_diffusion_loss_fn, _instance_lr_scheduler,
                          data_parallel_wrapper)


class DiffusionRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        autoencoder = kwargs.pop("autoencoder").eval()
        self.model = _instance_diffusion_model(autoencoder,
                                               self.args,
                                               self.device)
        if "cuda" in self.device:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model) 

        self.optimiser = _instance_optimiser(self.args, self.model)
        self.lr_scheduler = _instance_lr_scheduler(self.args, self.optimiser)
        self.loss_fn = _instance_diffusion_loss_fn(self.args)

        if self.args.sampling_only:
            self.temperature = self.args.sampling.temperature
            self.skip_steps = int(self.args.sampling.skip_steps)
        else:
            self.temperature = self.args.training.sampling_temperature
            self.skip_steps = int(self.args.training.sampling_skip_steps)

        # Wrap in accelerator classes
        (self.model,
        self.optimiser,
        self.lr_scheduler,
        self.train_loader,
        self.valid_loader,
        self.sample_loader) = self.accelerator.prepare(self.model,
                                 self.optimiser,
                                 self.lr_scheduler,
                                 self.train_loader,
                                 self.valid_loader,
                                 self.sample_loader)
        
        # Register objects for checkpointing
        objects_for_checkpointing = [
            self.model,
            self.optimiser,
            self.lr_scheduler
        ]
        objects_for_checkpointing = [obj_ckp for obj_ckp in objects_for_checkpointing if hasattr(obj_ckp, "state_dict") ]
        self.accelerator.register_for_checkpointing(*objects_for_checkpointing)

    def train_step(self, input, **kwargs):
        # Dictionary of outputs
        output = {}

        # Get condition from kwargs
        cond = kwargs.pop("condition", None)

        # Forward pass: predict model noise based on condition
        noise, noise_pred = self.model(input, cond)

        # Compute training loss
        loss = self.loss_fn(noise, noise_pred)
        
        # Zero grad and back propagation
        self.optimiser.zero_grad()

        self.accelerator.backward(loss)

        # Gradient Clipping
        if self.args.optim.grad_clip:
            self.accelerator.clip_grad_norm_(self.model.parameters(),
                                            self.args.optim.grad_clip)

        # Update gradients
        self.optimiser.step()

        # Update lr scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # Output dictionary update
        output.update({
            "loss": loss,
        })
        return output

    def valid_step(self, input, **kwargs):
        # Get condition from kwargs
        cond = kwargs.pop("condition", None)

        with torch.no_grad():
            # Forward pass: predict model noise based on condition
            noise, noise_pred = self.model(input, cond)

            # Compute validation loss
            loss = self.loss_fn(noise, noise_pred)

        # Output dictionary
        output = {
            "loss": loss,
        }
        return output

    def sample_step(self, input, **kwargs):
        # Get sampling parameters from kwargs
        cond = kwargs.pop("condition", None)

        with torch.no_grad():
            # One autoencoder forward pass to get shape of latent space -- can be optimised!
            z = self.model.module.ldm.autoencoder_encode(input, cond)

            # Sample latent space with diffusion model
            z_sample = self.model.module.sampler.sample(
                shape=z.shape,
                cond=cond,
                temperature=self.temperature,
                skip_steps=self.skip_steps,
                repeat_noise=False,
            )

            # Decode to reconstruct data
            sample = self.model.module.ldm.autoencoder_decode(z_sample, cond)
        return sample, z
