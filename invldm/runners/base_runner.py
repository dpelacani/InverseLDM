import os
import time
import fnmatch

import torch
import numpy as np
import platform
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Optional

from ..utils.utils import namespace2dict, gpu_diagnostics
from ..utils.visualisation import visualise_samples

torch.set_printoptions(sci_mode=False)


class BaseRunner(ABC):
    def __init__(self, **kwargs) -> None:

        self.args = kwargs.pop("args")
        self.logging_args = kwargs.pop("args_logging")
        self.run_args = kwargs.pop("args_run")

        self.sys_logger = self.logging_args.logger
        self.accelerator = self.run_args.accelerator

        self.train_loader = kwargs.pop("train_loader", None)
        self.valid_loader = kwargs.pop("valid_loader", None)
        self.sample_loader = kwargs.pop("sample_loader", None)

        self.device = self.run_args.device
        self.gpu_ids = self.run_args.gpu_ids

        self.model = None
        self.optimiser = None
        self.loss_fn = None
        self.lr_scheduler = None

        self.epoch = 1
        self.steps = torch.tensor([1]).to(self.accelerator.device)
        self.global_steps = -1

        self.init_steps = 0
        self.n_steps = self.get_total_round_steps()

        self.hparam_dict = {}

        self.completed = False



    @abstractmethod
    def train_step(self, input: torch.Tensor, **kwargs) -> dict:
        """ Function performs one batch training step. Input is a batch for training """
        return NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def valid_step(self, input: torch.Tensor, **kwargs) -> dict:
        """ Function performs one batch validation step. Input is a batch for validating """
        return NotImplementedError

    @torch.no_grad()
    def sample_step(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Function performs sampling of self.model."""
        return NotImplementedError

    def _update_hparam_dict(self) -> dict:
        hparam_dict = {
            "name": self.args.name,
            "dataset": self.train_loader.dataset.dataset.__dict__,
            "device": self.device,
            "device_ids": self.run_args.gpu_ids,
            "gpus": [torch.cuda.get_device_name(id) for id in self.run_args.gpu_ids],
            "processor": platform.machine() + " " + platform.processor() + " " + platform.system(),
            "seed": self.run_args.seed,
        }
        hparam_dict.update({"model": namespace2dict(self.args, flatten=True)})
        self.hparam_dict = hparam_dict
        return None

    def save_checkpoint(self, path: str = "") -> None:
        states = {
                "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimiser_state_dict": self.optimiser.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler,
                "epoch": self.epoch,
                "step": self.global_steps,
            }
        if self.args.name.lower() == "autoencoder" and self.args.model.adversarial_loss:
            states.update({
                "d_model_state_dict": self.accelerator.unwrap_model(self.d_model).state_dict(),
                "d_optimiser_state_dict": self.d_optimiser.optimizer.state_dict(),
                "d_lr_scheduler": self.d_lr_scheduler,
            })

        if not path:
            latest_ckpt_name = f"{self.args.name.lower()}_ckpt_latest.pth"
            path = os.path.join(self.args.ckpt_path, latest_ckpt_name)
        
        self.accelerator.save(states, path)
        self.sys_logger.info(f"Saved {self.args.name.lower()} checkpoint {path}")
        return None

    def load_checkpoint(self, path: Optional[str] = "", model_only: Optional[bool] = False) -> None:
        if not path :
            try:
                latest_ckpt_name = [f for f in os.listdir(self.args.ckpt_path) if fnmatch.fnmatch(f, "*latest*")][0]
                path = os.path.join(self.args.ckpt_path, latest_ckpt_name)
            except IndexError as e:
                if self.args.sampling_only:
                    self.sys_logger.critical(f"Could not find latest {self.args.name.lower()} model. Please specify checkpoint path to load.")
                    raise IndexError(e)
                else:
                    if self.run_args.y:
                        return None
                    else:
                        if self.args.training.n_epochs > 0:
                            self.sys_logger.critical(f"Could not find latest {self.args.name.lower()} model.")
                            user_input = input("\tProceed from scratch? (Y/N): ")
                            if user_input.lower() == "y" or user_input.lower() == "yes":
                                return None
                            else:
                                self.sys_logger.critical(f"Aborting...")
                                raise IndexError(e)
                        else:
                            return None

        self.sys_logger.info(f"Loading {self.args.name} checkpoint {path} ...")

        try:
            states = torch.load(path)
        except RuntimeError:
            states = torch.load(path, map_location="cpu")
        
        self.model.load_state_dict(states["model_state_dict"])

        if self.args.name == "autoencoder" and self.args.model.adversarial_loss:
            self.d_model.load_state_dict(states["d_model_state_dict"])
        
        if not model_only:
            self.optimiser.load_state_dict(states["optimiser_state_dict"])
            self.lr_scheduler = states["lr_scheduler"]
            self.epoch = states["epoch"]
            self.steps = torch.tensor([states["step"]]).to(self.accelerator.device)

            if self.args.name.lower() == "autoencoder" and self.args.model.adversarial_loss:
                self.d_optimiser.load_state_dict(states["d_optimiser_state_dict"])
                self.d_lr_scheduler = states["d_lr_scheduler"]


        self.sys_logger.info(f"{self.args.name.lower().capitalize()} checkpoint successfully loaded.")
        return None
    
    def get_checkpoint_path(self) -> str:
        # Initialise checkpoint path as None (find latest model)
        ckpt = None

        # Try to find checkpoint file if it's passed as a step integer
        if self.args.model.checkpoint is not None:
            try:
                ckpt_no = int(self.args.model.checkpoint)
                try:
                    ckpt_file = [f for f in os.listdir(self.args.ckpt_path) if fnmatch.fnmatch(f, f"*_step_{ckpt_no}*")][0]
                    ckpt = os.path.join(self.args.ckpt_path, ckpt_file)
                except IndexError:
                    self.sys_logger.critical(f"Tried to load step {ckpt_no} from {self.args.ckpt_path} but found no file. Loading latest model...")
            except ValueError:
                ckpt = self.args.model.checkpoint
        return ckpt
    
    def resume_training(self) -> None:
        # Load the checkpoint
        ckpt = self.get_checkpoint_path()
        self.load_checkpoint(ckpt)

        # Prevent running train if it's complete
        if self.epoch >= self.args.training.n_epochs:
            self.sys_logger.info(f"{self.args.name.lower().capitalize()} training already completed with {self.epoch} epochs")
            self.completed = True

        # Resume training at beginning of current epoch
        self.steps = torch.tensor([int((self.epoch - 1) * len(self.train_loader)) + 1]).to(self.accelerator.device)

        # Store the initial steps (used for progression logs)
        self.init_steps = self.steps.cpu().item()
        self.n_steps = self.get_total_round_steps()
        return None

    def save_figure(self, x: torch.Tensor, step: int, mode: str, fig_type: str, save_tensor: Optional[bool] = False
                    , scale: Optional[bool] = True) -> None:
        assert mode in ["training", "validation", "sampling", ""]

        if self.args.sampling_only:
            file_name = f"{self.args.name.lower()}_{mode}_{fig_type}_batch_{step}.png"
            path = os.path.join(self.run_args.samples_folder, file_name)

        else:
            file_name = f"{self.args.name.lower()}_{mode}_{fig_type}_epoch_{self.epoch}_step_{step}.png"
            if fig_type in ["recon", "input", "error"]:
                path = os.path.join(self.args.recon_path, file_name)
            elif fig_type in ["sample", "sample_input", "sample_recon", "sample_error"]:
                path = os.path.join(self.args.samples_path, file_name)
            else:
                path = os.path.join(self.args.log_path, file_name)
            
        fig = visualise_samples(x.cpu(), scale=scale)
        plt.savefig(path)
        plt.close(fig)
        self.sys_logger.info(f"Saved {self.args.name} {mode} {fig_type} figure in {path}", main_process_only=False, in_order=True)

        if save_tensor:
            path = path[:-3] + "pt"
            torch.save(x, path)
            self.sys_logger.info(f"Saved {self.args.name} {mode} {fig_type} tensor in {path}", main_process_only=False, in_order=True)
        return None

    def get_total_round_steps(self) -> int:
        if self.args.sampling_only:
            if self.sample_loader is not None:
                return len(self.sample_loader)
            else:
                return -1
        else:
            return int((self.args.training.n_epochs) * len(self.train_loader)) - self.init_steps

    def progress(self) -> float:
        return (self.global_steps - self.init_steps) / self.n_steps

    def eta(self, start_time: float) -> str:
        try:
            progress = self.progress()
            current_time = np.round((time.time() - start_time) / 60, 2)
            expected_total_time = np.round(current_time * (self.n_steps / (self.global_steps - self.init_steps)), 2)
            if self.n_steps >= 10:
                eta = np.round(expected_total_time - current_time, 2)
                h = int(np.floor(eta / 60))
                m = int(np.floor(eta - h*60))
                s = int(np.floor((eta - h*60 - m)*60))
            else:
                h, m, s = 99, 99, 99
            s = "Progress: {:.2f}%  ETA: {:02d}:{:02d}:{:02d}".format(progress*100, h, m, s)
            return s
        except (RuntimeWarning, ZeroDivisionError, OverflowError, ValueError):
            return ""

    def train(self) -> None:
        assert (not self.args.sampling_only), " Sampling only flags cannot be passed for training experiment. "

        start_time = time.time()

        # Prepare iterator for validation
        try:
            valid_iterator = iter(self.valid_loader)
        except TypeError:
            valid_iterator = None

        # Adjust start epoch, load checkpoint
        if self.run_args.resume_training or self.args.model.checkpoint:
            self.resume_training()
        start_epoch = self.epoch

        # Fetch tracking logger
        logger = self.logging_args.tracker

        # Update hyperparameter dictionary
        self._update_hparam_dict()

        # Training mode
        self.model.train()


        # Loop through batch and perform training, validation and sampling steps
        if not self.completed:
            try:
                for epoch in range(start_epoch, self.args.training.n_epochs + 1):
                    for i, batch in enumerate(self.train_loader):
                        
                        # Update global step
                        self.global_steps = self.accelerator.reduce(self.steps).item() - (self.accelerator.num_processes - self.accelerator.process_index) + 1
                        
                        # Unpack batch into input and condition (if returned) and send to device
                        if isinstance(batch, list):
                            input, condition = batch
                            input, condition = input.float(), condition.float()
                        else:
                            input, condition = batch, None
                            input = input.float()

                        # Train batch and get loss
                        output = self.train_step(input, condition=condition)
                        try:
                            loss = output["loss"]
                        except KeyError as e:
                            self.sys_logger.critical("Train step output must be a dictionary containing the key 'loss'")
                            raise KeyError(e)
                        self.sys_logger.info(f"> Process {self.accelerator.process_index}: {self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.global_steps}/{self.n_steps} Training Loss: {loss.item()} {self.eta(start_time)}", main_process_only=False, in_order=True)
                        
                        # Get discriminator Loss if exists
                        if "loss_d" in output.keys():
                            loss_d = output["loss_d"]
                            self.sys_logger.info(f"> Process {self.accelerator.process_index}: {self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.global_steps}/{self.n_steps} Discriminator Training Loss: {loss_d.item()} {self.eta(start_time)}", main_process_only=False, in_order=True)

                        # Log loss in logger
                        if logger:    
                            gathered_losses = self.accelerator.gather(loss)
                            if "loss_d" in output.keys():
                                gathered_losses_d = self.accelerator.gather(loss)
                                
                            if self.accelerator.is_main_process:
                                for i in range(len(gathered_losses)):
                                    step = self.global_steps + i
                                    logger.log_scalar(
                                        tag=f"{self.args.name.lower()}_training_loss",
                                        val=gathered_losses[i].item(),
                                        step=step
                                    )
                                    logger.log_hparams(
                                        hparam_dict=self.hparam_dict,
                                        metric_dict={'hparam/train_loss': gathered_losses.mean().item()}
                                    )
                                    if self.lr_scheduler:
                                        logger.log_scalar(
                                            tag=f"{self.args.name.lower()}_optim_lr",
                                            val=torch.tensor(self.lr_scheduler.get_last_lr()).mean().item(),
                                            step=step
                                        )
                                    if "gathered_losses_d" in locals():
                                        logger.log_scalar(
                                            tag=f"{self.args.name.lower()}_discriminator_training_loss",
                                            val=gathered_losses_d[i].item(),
                                            step=step
                                        )


                        # Save training recon fig
                        try:
                            if self.args.training.save_recon_freq > 0 and self.global_steps % self.args.training.save_recon_freq == 0:
                                recon = output["recon"]
                                error = input-recon
                                
                                self.save_figure(input, self.global_steps, "training", "input")
                                self.save_figure(recon, self.global_steps, "training", "recon")
                                self.save_figure(error / torch.max(torch.abs(error)), self.global_steps, "training", "error", scale=False)

                                # if logger:
                                #     gathered_input, gathered_recon = self.accelerator.gather(input), self.accelerator.gather(recon)
                                #     gathered_error = self.accelerator.gather(error)
                                #     if self.accelerator.is_main_process:
                                #         for i in range(len(gathered_input)): 
                                #             step = self.global_steps + i
                                #             fig = visualise_samples(gathered_input[i], scale=True)
                                #             logger.log_figure(
                                #                 tag=f"{self.args.name.lower()}_training_input",
                                #                 fig=fig,
                                #                 step=step
                                #             )
                                #             plt.close(fig)
                                #             logger.log_figure(
                                #                 tag=f"{self.args.name.lower()}_training_recon",
                                #                 fig=visualise_samples(gathered_recon[i], scale=True),
                                #                 step=step
                                #             )
                                #             logger.log_figure(
                                #                 tag=f"{self.args.name.lower()}_training_error",
                                #                 fig=visualise_samples(gathered_error[i]/torch.max(torch.abs(gathered_error[i])), scale=False),
                                #                 step=step
                                #             )
                        except (KeyError, AttributeError):
                            pass

                        # # Memory diagnostics:
                        # # gpu_diagnostics()

                        # Sample and save training figure
                        if self.args.training.sampling_freq > 0 and self.global_steps % self.args.training.sampling_freq == 0:
                            try:
                                sample, _ = self.sample_step(
                                    torch.randn_like(input),
                                    condition = condition,
                                )

                                # If condition is None, sample is a randomly generated model from the learned distribution.
                                # Since autoencoder is not conditioned, save sample as if condition is None.
                                if condition is None or self.args.name == "autoencoder":
                                    self.save_figure(sample, self.global_steps, "training", "sample")

                                    # if logger:
                                    #     logger.log_figure(
                                    #         tag=f"{self.args.name.lower()}_training_sample",
                                    #         fig=visualise_samples(sample, scale=True),
                                    #         step=self.global_steps
                                    #     )
                                # If condition is passed, we want to compare the conditionally sampled model to the input
                                else:
                                    self.save_figure(input, self.global_steps, "training", "sample_input")
                                    self.save_figure(sample, self.global_steps, "training", "sample_recon")
                                    self.save_figure(input - sample, self.global_steps, "training", "sample_error", scale=False)

                                    # if logger:
                                    #     logger.log_figure(
                                    #         tag=f"{self.args.name.lower()}_training_sample_input",
                                    #         fig=visualise_samples(input, scale=True),
                                    #         step=self.steps
                                    #     )
                                    #     logger.log_figure(
                                    #         tag=f"{self.args.name.lower()}_training_sample_recon",
                                    #         fig=visualise_samples(sample, scale=True),
                                    #         step=self.steps
                                    #     )
                                    #     logger.log_figure(
                                    #         tag=f"{self.args.name.lower()}_training_sample_error",
                                    #         fig=visualise_samples(input - sample, scale=False),
                                    #         step=self.steps
                                    #     )

                            except NotImplementedError:
                                pass


                        # # Validate batch
                        # if valid_iterator:
                        #     # Check which validation steps to run (validate, save_recon and sampling)
                        #     valid_freq = self.args.validation.freq > 0 and self.global_steps%self.args.validation.freq == 0
                        #     try:
                        #         valid_save_recon_freq = self.args.validation.save_recon_freq > 0 and self.global_steps%self.args.validation.save_recon_freq == 0
                        #     except AttributeError:
                        #         valid_save_recon_freq = False
                        #         pass
                        #     try:
                        #         valid_sampling_frequency = self.args.validation.sampling_freq> 0 and self.global_steps%self.args.validation.sampling_freq== 0
                        #     except AttributeError:
                        #         valid_sampling_frequency = False

                        #     # If any validation step required, validate batch
                        #     if valid_freq or valid_save_recon_freq or valid_sampling_frequency:
                        #         tt_val_loss, tt_val_loss_d = 0., 0.
                        #         for j, val_batch in enumerate(self.valid_loader):
                        #             # try:

                        #             # Unpack validation batch into input and condition (if returned) and send to device.
                        #             if isinstance(val_batch, list):
                        #                 val_input, val_condition = val_batch
                        #                 val_input, val_condition = val_input.float(), val_condition.float()
                        #             else:
                        #                 val_input, val_condition = val_batch, None
                        #                 val_input = val_input.float()
                                    
                        #             # # Restart iterator and get batch
                        #             # except StopIteration:
                        #             #     valid_iterator = iter(self.valid_loader)  
                        #             #     if isinstance(val_batch, list):
                        #             #         val_input, val_condition = val_batch
                        #             #         val_input, val_condition = val_input.float(), val_condition.float()
                        #             #     else:
                        #             #         val_input, val_condition = val_batch, None
                        #             #         val_input = val_input.float()

                        #             # Validate batch
                        #             val_output = self.valid_step(val_input, condition=val_condition)

                        #             try:
                        #                 val_loss = val_output["loss"]
                        #                 tt_val_loss += val_loss.item()
                        #             except KeyError as e:
                        #                 self.sys_logger.critical("Validation step output must be a dictionary containing the key 'loss'")
                        #                 raise KeyError(e)
                                    
                        #             if "loss_d" in output.keys():
                        #                 val_loss_d = output["loss_d"]
                        #                 tt_val_loss_d += val_loss_d.item()

                        #         val_loss = tt_val_loss / len(self.valid_loader)                                    
                        #         self.sys_logger.info(f"> Process {self.accelerator.process_index}: {self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.global_steps}/{self.n_steps} Validation Loss: {val_loss.item()}", main_process_only=False, in_order=True)

                        #         if tt_val_loss_d in locals():
                        #             val_loss_d = tt_val_loss_d / len(self.valid_loader)
                        #             self.sys_logger.info(f"> Process {self.accelerator.process_index}: {self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.global_steps}/{self.n_steps} Discriminator Validation Loss: {val_loss_d.item()}", main_process_only=False, in_order=True)

                        #         # Log validation
                        #         if logger:

                        #             logger.log_scalar(
                        #                 tag=f"{self.args.name.lower()}_valid_loss",
                        #                 val=val_loss.item(),
                        #                 step=self.steps
                        #             )
                        #             logger.log_hparams(
                        #                 hparam_dict=self.hparam_dict,
                        #                 metric_dict={'hparam/valid_loss': val_loss.item()}
                        #             )
                        #             if "loss_d" in output.keys():
                        #                 logger.log_scalar(
                        #                     tag=f"{self.args.name.lower()}_discriminator_valid_loss",
                        #                     val=val_loss_d.item(),
                        #                     step=self.steps
                        #                 )

                        #         # Save validation recon figure
                        #         if valid_save_recon_freq:
                        #             try:
                        #                 val_recon = val_output["recon"]
                        #                 val_error = val_input - val_recon
                        #                 self.save_figure(val_input, self.global_steps, "validation", "input")
                        #                 self.save_figure(val_recon, self.global_steps, "validation", "recon")
                        #                 self.save_figure(val_error/torch.max(torch.abs(val_error)), self.global_steps, "validation", "error", scale=False)
                        #         #         logger.log_figure(
                        #         #             tag=f"{self.args.name.lower()}_valid_input",
                        #         #             fig=visualise_samples(val_input, scale=True),
                        #         #             step=self.steps
                        #         #         )
                        #         #         logger.log_figure(
                        #         #             tag=f"{self.args.name.lower()}_valid_recon",
                        #         #             fig=visualise_samples(val_recon, scale=True),
                        #         #             step=self.steps
                        #         #         )
                        #         #         logger.log_figure(
                        #         #             tag=f"{self.args.name.lower()}_valid_error",
                        #         #             fig=visualise_samples(val_input - val_recon, scale=False),
                        #         #             step=self.steps
                        #         #         )
                        #             except KeyError:
                        #                 pass

                        #         # Sample and save validation if condition is passed
                        #         if valid_sampling_frequency and condition is not None and self.args.name != "autoencoder":
                        #             try:
                        #                 val_sample, _ = self.sample_step(
                        #                     torch.randn_like(val_input),
                        #                     condition = val_condition,
                        #                 )
                                        
                        #                 self.save_figure(val_input,  self.global_steps, "validation", "sample_input")
                        #                 self.save_figure(val_sample, self.global_steps,  "validation", "sample_recon")
                        #                 self.save_figure(val_input - val_sample, self.global_steps,  "validation", "sample_error", scale=False)

                        #         #         if logger:
                        #         #             logger.log_figure(
                        #         #                 tag=f"{self.args.name.lower()}_valid_sample_input",
                        #         #                 fig=visualise_samples(val_input, scale=True),
                        #         #                 step=self.steps
                        #         #             )
                        #         #             logger.log_figure(
                        #         #                 tag=f"{self.args.name.lower()}_valid_sample_recon",
                        #         #                 fig=visualise_samples(val_sample, scale=True),
                        #         #                 step=self.steps
                        #         #             )
                        #         #             logger.log_figure(
                        #         #                 tag=f"{self.args.name.lower()}_valid_sample_error",
                        #         #                 fig=visualise_samples(val_input - val_sample, scale=False),
                        #         #                 step=self.steps
                        #         #             )

                        #             except NotImplementedError:
                        #                 pass

                        #         # Memory diagnostics
                        #         # gpu_diagnostics()

                        # # Save training checkpoints
                        if (self.args.training.ckpt_freq > 0 and self.global_steps % self.args.training.ckpt_freq == 0) or epoch == self.args.training.n_epochs:
                            if not self.args.training.ckpt_last_only:
                                ckpt_name = f"{self.args.name.lower()}_ckpt_epoch_{self.epoch}_step_{self.global_steps}.pth"
                                ckpt_path = os.path.join(self.args.ckpt_path, ckpt_name)
                                self.save_checkpoint(ckpt_path)

                            # Update latest checkpoint
                            self.save_checkpoint()

                        self.steps += 1

                    self.epoch += 1

            except KeyboardInterrupt:
                self.save_checkpoint()
                raise (KeyboardInterrupt)
        self.completed = True
        return None

    # @torch.no_grad()
    # def validate(self) -> None:
    #     if self.valid_loader:
    #         total_loss = 0.
    #         for i, batch in enumerate(self.valid_loader):
    #             # Unpack batch into input and condition (if returned) and send to device
    #             if isinstance(batch, list):
    #                 input, condition = batch
    #                 input, condition = input.float(), condition.float()
    #                 # input, condition = input.float().to(self.device), condition.float().to(self.device)
    #             else:
    #                 input, condition = batch, None
    #                 input = input.float()
    #                 # input = input.float().to(self.device)

    #              # Validation step   
    #             output = self.valid_step(input, condition=condition)
    #             try:
    #                 loss = output["loss"]
    #             except KeyError as e:
    #                 self.sys_logger.critical("Validation step output must be a dictionary containing the key 'loss'")
    #                 raise KeyError(e)
    #             total_loss += loss * input.shape[0]

    #         # Average validation losses
    #         avg_loss = total_loss / len(self.valid_loader.dataset)
    #         self.sys_logger.info(f"{self.args.name.lower().capitalize()} Average validation loss: {avg_loss.item()}")
    #     return None

    # @torch.no_grad()
    # def sample(self) -> None:
    #     start_time = time.time()
    #     if self.sample_loader:
    #         for i, sample_batch in enumerate(self.sample_loader):      
    #             # Unpack sampling batch into input and condition (if returned) and send to device.          
    #             if isinstance(sample_batch, list):
    #                 input, condition = sample_batch
    #                 input, condition = input.float(), condition.float()
    #                 # input, condition = input.float().to(self.device), condition.float().to(self.device)
    #             else:
    #                 input, condition = sample_batch, None
    #                 input = input.float()
    #                 # input = input.float().to(self.device)

    #             sample, _ = self.sample_step(input, condition=condition)
    #             self.save_figure(sample, "", "sample", save_tensor=True)

    #             self.sys_logger.info(f"{self.args.name.lower().capitalize()} Sampling batch {self.steps} / {len(self.sample_loader)} concluded. {self.eta(start_time)}")

    #             if self.args.sampling_only:
    #                 self.steps += 1
    #     return None

