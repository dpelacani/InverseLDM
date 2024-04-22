import os

import torch
import torch.nn as nn

import accelerate

from . import BaseRunner
from ..models.utils import (_instance_autoencoder_model, _instance_optimiser,
                          _instance_autoencoder_loss_fn, _instance_lr_scheduler,
                          _instance_discriminator_model, _instance_discriminator_loss_fn,
                          data_parallel_wrapper, set_requires_grad)


class AutoencoderRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Instantiate autoencoder, use SyncBatchNorm for multi gpu
        self.model = _instance_autoencoder_model(self.args, self.device)
        if "cuda" in self.device:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model) 

        # Initialise as None to prevent errors
        self.d_model, self.d_optimiser, self.d_lr_scheduler = None, None, None

        # If not in sampling only mode, instantiate optimising objects
        if not self.args.sampling_only: 
            assert (self.train_loader is not None), " Train data loader is required in training mode, but got None"
            self.optimiser = _instance_optimiser(self.args, self.model)
            self.lr_scheduler = _instance_lr_scheduler(self.args, self.optimiser)
            self.loss_fn = _instance_autoencoder_loss_fn(self.args)

            # Instantiate optimising objects for adversarial loss
            if self.args.model.adversarial_loss:
                self.d_model = _instance_discriminator_model(self.args, self.device)
                if "cuda" in self.device:
                    self.d_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model)
                self.d_optimiser = _instance_optimiser(self.args, self.d_model)
                self.d_lr_scheduler = _instance_lr_scheduler(self.args, self.d_optimiser)
                self.d_loss_fn = _instance_discriminator_loss_fn(self.args)
            

        # Wrap in accelerator classes
        (self.model,
        self.optimiser,
        self.lr_scheduler,
        self.d_model,
        self.d_optimiser,
        self.d_lr_scheduler,
        self.train_loader,
        self.valid_loader,
        self.sample_loader) = self.accelerator.prepare(self.model,
                                 self.optimiser,
                                 self.lr_scheduler,
                                 self.d_model,
                                 self.d_optimiser,
                                 self.d_lr_scheduler,
                                 self.train_loader,
                                 self.valid_loader,
                                 self.sample_loader)
        
        # Register objects for checkpointing
        objects_for_checkpointing = [
            self.model,
            self.optimiser,
            self.lr_scheduler,
            self.d_model,
            self.d_optimiser,
            self.d_lr_scheduler]
        objects_for_checkpointing = [obj_ckp for obj_ckp in objects_for_checkpointing if hasattr(obj_ckp, "state_dict") ]
        self.accelerator.register_for_checkpointing(*objects_for_checkpointing)
        

    def train_step(self, input, **kwargs):
        # Get condition from kwargs
        cond = kwargs.pop("condition", None)

        # Forward pass: recon and the statistical posterior
        recon, mean, log_var = self.model(input, cond)

        # Compute training loss
        loss = self.loss_fn(input, recon, mean, log_var)

        # Discriminator loss (train generator)
        if self.args.model.adversarial_loss:
            # Disable grad for discriminator
            set_requires_grad(self.d_model, requires_grad=False)
            logits_fake = self.d_model(recon.contiguous())
            loss += self.d_loss_fn(logits_fake, is_real=True, apply_weight=True) # fool discriminator

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

        # Discriminator loss (train discriminator)
        loss_d = torch.tensor(-1.)
        if self.args.model.adversarial_loss:
            # Enable grad for discriminator
            set_requires_grad(self.d_model, requires_grad=True)
            
            # Get predictions
            logits_true = self.d_model(input.contiguous())
            logits_fake = self.d_model(recon.detach().contiguous())

            # Compute loss
            loss_d = 0.5 * (self.d_loss_fn(logits_fake, is_real=False) + self.d_loss_fn(logits_true, is_real=True))

            # Zero grad and back propagation
            self.d_optimiser.zero_grad()
            self.accelerator.backward(loss_d)

            # Update gradients
            self.d_optimiser.step()

            # Update lr scheduler
            if self.d_lr_scheduler:
                self.d_lr_scheduler.step()

        # Output dictionary
        output = {
            "loss": loss,
            "recon": recon,
            "mean": mean,
            "log_var": log_var,
            "loss_d": loss_d,
        }
        return output

    def valid_step(self, input, **kwargs):
        with torch.no_grad():
            # Forward pass: recon and the statistical posterior
            recon, mean, log_var = self.model(input)

            # Compute validation loss
            loss = self.loss_fn(input, recon, mean, log_var)

            # Compute validation loss for discriminator
            loss_d = torch.tensor(-1.)
            if self.args.model.adversarial_loss:
                with torch.no_grad():
                    # Get predictions
                    logits_true = self.d_model(input.contiguous())
                    logits_fake = self.d_model(recon.contiguous())

                    # Compute loss
                    loss_d = 0.5 * (self.d_loss_fn(logits_fake, is_real=False) + self.d_loss_fn(logits_true, is_real=True))

        # Output dictionary
        output = {
            "loss": loss,
            "recon": recon,
            "mean": mean,
            "log_var": log_var,
            "loss_d": loss_d,
        }
        return output
    
    def sample_step(self, input, **kwargs):
        with torch.no_grad():
            _, _ = self.model.module.model.encode(input)
            z = self.model.module.model.sample()

            sample = self.model.module.model.decode(z)
        return sample, z      


