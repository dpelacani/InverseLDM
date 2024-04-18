import torch
import ray

from . import BaseRunner
from ..models.utils import (_instance_autoencoder_model, _instance_optimiser,
                          _instance_autoencoder_loss_fn, _instance_lr_scheduler,
                          _instance_discriminator_model, _instance_discriminator_loss_fn,
                          data_parallel_wrapper, set_requires_grad)


class AutoencoderRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = _instance_autoencoder_model(self.args, self.device)
        # self.model = data_parallel_wrapper(module=self.model,
        #                                    device=self.device,
        #                                    device_ids=self.gpu_ids)
        # self.device = self.model.module.device
        # self.model = ray.train.torch.prepare_model(self.model)

        # If not in sampling only mode, instantiate optimising objects
        if not self.args.sampling_only: 
            assert (self.train_loader is not None), " Train data loader is required in training mode, but got None"
            self.optimiser = _instance_optimiser(self.args, self.model)
            self.lr_scheduler = _instance_lr_scheduler(self.args, self.optimiser)
            self.loss_fn = _instance_autoencoder_loss_fn(self.args)

            # Instantiate optimising objects for adversarial loss
            if self.args.model.adversarial_loss:
                self.d_model = _instance_discriminator_model(self.args, self.device)
                # self.d_model = data_parallel_wrapper(module=self.d_model,
                #                                     device=self.device,
                #                                     device_ids=self.gpu_ids)
                # self.d_model = ray.train.torch.prepare_model(self.d_model)
                self.d_optimiser = _instance_optimiser(self.args, self.d_model)
                self.d_lr_scheduler = _instance_lr_scheduler(self.args, self.d_optimiser)
                self.d_loss_fn = _instance_discriminator_loss_fn(self.args)

    def train_step(self, input, **kwargs):
        torch.autograd.set_detect_anomaly(True)
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
        loss.backward()

        # Gradient Clipping
        if self.args.optim.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
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
            loss_d.backward()

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
        # Forward pass: recon and the statistical posterior
        self.model.eval()
        recon, mean, log_var = self.model(input)

        # Compute validation loss
        loss = self.loss_fn(input, recon, mean, log_var)

        # Compute validation loss for discriminator
        loss_d = torch.tensor(-1.)
        if self.args.model.adversarial_loss:
            self.d_model.eval()
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
        _, _ = self.model.module.model.encode(input)
        z = self.model.module.model.sample()

        sample = self.model.module.model.decode(z)
        return sample, z    


