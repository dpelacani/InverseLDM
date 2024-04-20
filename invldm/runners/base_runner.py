import os
import time
import fnmatch

import torch
import ray
import numpy as np
import logging
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
        self.steps = 1

        self.init_steps = 0
        self.n_steps = self.get_total_round_steps()

        self.hparam_dict = {}

        self.completed = False

        self.sys_logger = self.logging_args.logger

        # self.global_step_actor = GlobalStepActor.remote()

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
            "device_ids": self.model.device_ids if isinstance(self.model, torch.nn.parallel.distributed.DistributedDataParallel) else None,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if "cuda" in self.device else [],
            "processor": platform.machine() + " " + platform.processor() + " " + platform.system(),
            "ray_num_workers": self.run_args.ray_num_workers,
            "seed": self.run_args.seed,
        }
        hparam_dict.update({"model": namespace2dict(self.args, flatten=True)})
        self.hparam_dict = hparam_dict
        return None

    def save_checkpoint(self, path: str = "") -> None:
        states = {
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "lr_scheduler": self.lr_scheduler,
                "epoch": self.epoch,
                "step": self.steps,
            }
        if self.args.name.lower() == "autoencoder" and self.args.model.adversarial_loss:
            states.update({
                "d_model_state_dict": self.d_model.state_dict(),
                "d_optimiser_state_dict": self.d_optimiser.state_dict(),
                "d_lr_scheduler": self.d_lr_scheduler,
            })
        if not path:
            latest_ckpt_name = f"{self.args.name.lower()}_ckpt_latest.pth"
            path = os.path.join(self.args.ckpt_path, latest_ckpt_name)

        torch.save(states, path)
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
            self.steps = states["step"]

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
        self.steps = int((self.epoch - 1) * len(self.train_loader)) + 1

        # Store the initial steps (used for progression logs)
        self.init_steps = self.steps
        self.n_steps = self.get_total_round_steps()
        return None

    def save_figure(self, x: torch.Tensor, mode: str, fig_type: str, save_tensor: Optional[bool] = False
                    , scale: Optional[bool] = True) -> None:
        assert mode in ["training", "validation", "sampling", ""]

        if self.args.sampling_only:
            file_name = f"{self.args.name.lower()}_{mode}_{fig_type}_batch_{self.steps}.png"
            path = os.path.join(self.run_args.samples_folder, file_name)

        else:
            file_name = f"{self.args.name.lower()}_{mode}_{fig_type}_epoch_{self.epoch}_step_{self.steps}.png"
            if fig_type in ["recon", "input", "error"]:
                path = os.path.join(self.args.recon_path, file_name)
            elif fig_type in ["sample", "sample_input", "sample_recon", "sample_error"]:
                path = os.path.join(self.args.samples_path, file_name)
            else:
                path = os.path.join(self.args.log_path, file_name)
            
        fig = visualise_samples(x, scale=scale)
        plt.savefig(path)
        plt.close(fig)
        self.sys_logger.info(f"Saved {self.args.name} {mode} {fig_type} figure in {path}")

        if save_tensor:
            path = path[:-3] + "pt"
            torch.save(x, path)
            self.sys_logger.info(f"Saved {self.args.name} {mode} {fig_type} tensor in {path}")
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
        return (self.steps - self.init_steps) / self.n_steps

    def eta(self, start_time: float) -> str:
        try:
            progress = self.progress()
            current_time = np.round((time.time() - start_time) / 60, 2)
            expected_total_time = np.round(current_time * (self.n_steps / (self.steps - self.init_steps)), 2)
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
                    
                    if ray.train.get_context().get_world_size() > 1:
                        self.train_loader.sampler.set_epoch(epoch)

                    for i, batch in enumerate(self.train_loader):

                        # Unpack batch into input and condition (if returned) and send to device
                        if isinstance(batch, list):
                            input, condition = batch
                            # input, condition = input.float().to(self.device), condition.float().to(self.device)
                            input, condition = input.float(), condition.float()
                        else:
                            input, condition = batch, None
                            # input = input.float().to(self.device)
                            input = input.float()

                        # Train batch
                        output = self.train_step(input, condition=condition)
                        try:
                            loss = output["loss"]
                        except KeyError as e:
                            self.sys_logger.critical("Train step output must be a dictionary containing the key 'loss'")
                            raise KeyError(e)

                        # Log loss
                        self.sys_logger.info(f"{self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.steps} Training Loss: {loss.item()} {self.eta(start_time)}")
                        if logger:
                            logger.log_scalar(
                                tag=f"{self.args.name.lower()}_training_loss",
                                val=loss.item(),
                                step=self.steps
                            )
                            logger.log_hparams(
                                hparam_dict=self.hparam_dict,
                                metric_dict={'hparam/train_loss': loss.item()}
                            )
                            if self.lr_scheduler:
                                logger.log_scalar(
                                    tag=f"{self.args.name.lower()}_optim_lr",
                                    val=torch.tensor(self.lr_scheduler.get_last_lr()).mean().item(),
                                    step=self.steps
                                )
                            if "loss_d" in output.keys():
                                loss_d = output["loss_d"]
                                logger.log_scalar(
                                    tag=f"{self.args.name.lower()}_discriminator_training_loss",
                                    val=loss_d.item(),
                                    step=self.steps
                                )
                                self.sys_logger.info(f"{self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.steps} Discriminator Training Loss: {loss_d.item()} {self.eta(start_time)}")


                        # Save training recon fig
                        try:
                            if self.args.training.save_recon_freq > 0 and self.steps % self.args.training.save_recon_freq == 0:
                                recon = output["recon"]
                                self.save_figure(input, "training", "input")
                                self.save_figure(recon, "training", "recon")
                                self.save_figure(input-recon, "training", "error", scale=False)
                                if logger:
                                    logger.log_figure(
                                        tag=f"{self.args.name.lower()}_training_input",
                                        fig=visualise_samples(input, scale=True),
                                        step=self.steps
                                    )
                                    logger.log_figure(
                                        tag=f"{self.args.name.lower()}_training_recon",
                                        fig=visualise_samples(recon, scale=True),
                                        step=self.steps
                                    )
                                    logger.log_figure(
                                        tag=f"{self.args.name.lower()}_training_error",
                                        fig=visualise_samples(input-recon, scale=False),
                                        step=self.steps
                                    )
                        except (KeyError, AttributeError):
                            pass

                        # Memory diagnostics:
                        # gpu_diagnostics()

                        # Sample and save training figure
                        if self.args.training.sampling_freq > 0 and self.steps % self.args.training.sampling_freq == 0:
                            try:
                                sample, _ = self.sample_step(
                                    torch.randn_like(input),
                                    condition = condition,
                                )

                                # If condition is None, sample is a randomly generated model from the learned distribution.
                                # Since autoencoder is not conditioned, save sample as if condition is None.
                                if condition is None or self.args.name == "autoencoder":
                                    self.save_figure(sample, "training", "sample")

                                    if logger:
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_training_sample",
                                            fig=visualise_samples(sample, scale=True),
                                            step=self.steps
                                        )
                                # If condition is passed, we want to compare the conditionally sampled model to the input
                                else:
                                    self.save_figure(input, "training", "sample_input")
                                    self.save_figure(sample, "training", "sample_recon")
                                    self.save_figure(input - sample, "training", "sample_error", scale=False)

                                    if logger:
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_training_sample_input",
                                            fig=visualise_samples(input, scale=True),
                                            step=self.steps
                                        )
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_training_sample_recon",
                                            fig=visualise_samples(sample, scale=True),
                                            step=self.steps
                                        )
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_training_sample_error",
                                            fig=visualise_samples(input - sample, scale=False),
                                            step=self.steps
                                        )

                            except NotImplementedError:
                                pass


                        # Validate batch
                        if valid_iterator:
                            # Check which validation steps to run (validate, save_recon and sampling)
                            valid_freq = self.args.validation.freq > 0 and self.steps%self.args.validation.freq == 0
                            try:
                                valid_save_recon_freq = self.args.validation.save_recon_freq > 0 and self.steps%self.args.validation.save_recon_freq == 0
                            except AttributeError:
                                valid_save_recon_freq = False
                                pass
                            try:
                                valid_sampling_frequency = self.args.validation.sampling_freq> 0 and self.steps%self.args.validation.sampling_freq== 0
                            except AttributeError:
                                valid_sampling_frequency = False

                            # If any validation step required, validate batch
                            if valid_freq or valid_save_recon_freq or valid_sampling_frequency:
                                try:
                                    val_batch = next(valid_iterator)

                                    # Unpack validation batch into input and condition (if returned) and send to device.
                                    if isinstance(val_batch, list):
                                        val_input, val_condition = val_batch
                                        # val_input, val_condition = val_input.float().to(self.device), val_condition.float().to(self.device)
                                        val_input, val_condition = val_input.float(), val_condition.float()
                                    else:
                                        val_input, val_condition = val_batch, None
                                        # val_input = val_input.float().to(self.device)
                                        val_input = val_input.float()
                                
                                # Restart iterator and get batch
                                except StopIteration:
                                    valid_iterator = iter(self.valid_loader)  
                                    if isinstance(val_batch, list):
                                        val_input, val_condition = val_batch
                                        # val_input, val_condition = val_input.float().to(self.device), val_condition.float().to(self.device)
                                        val_input, val_condition = val_input.float(), val_condition.float()
                                    else:
                                        val_input, val_condition = val_batch, None
                                        # val_input = val_input.float().to(self.device)
                                        val_input = val_input.float()

                                # Validate batch
                                val_output = self.valid_step(val_input, condition=val_condition)
                                try:
                                    val_loss = val_output["loss"]
                                except KeyError as e:
                                    self.sys_logger.critical("Validation step output must be a dictionary containing the key 'loss'")
                                    raise KeyError(e)

                                # Log validation
                                self.sys_logger.info(f"{self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.steps} Validation Loss: {val_loss.item()}")
                                if logger:
                                    logger.log_scalar(
                                        tag=f"{self.args.name.lower()}_valid_loss",
                                        val=val_loss.item(),
                                        step=self.steps
                                    )
                                    logger.log_hparams(
                                        hparam_dict=self.hparam_dict,
                                        metric_dict={'hparam/valid_loss': val_loss.item()}
                                    )
                                    if "loss_d" in output.keys():
                                        logger.log_scalar(
                                            tag=f"{self.args.name.lower()}_discriminator_valid_loss",
                                            val=output["loss_d"].item(),
                                            step=self.steps
                                        )
                                        self.sys_logger.info(f"{self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.steps} Discriminator Validation Loss: {loss_d.item()} {self.eta(start_time)}")

                                # Save validation recon figure
                                if valid_save_recon_freq:
                                    try:
                                        val_recon = val_output["recon"]
                                        self.save_figure(val_input, "validation", "input")
                                        self.save_figure(val_recon, "validation", "recon")
                                        self.save_figure(val_input - val_recon, "validation", "error", scale=False)
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_valid_input",
                                            fig=visualise_samples(val_input, scale=True),
                                            step=self.steps
                                        )
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_valid_recon",
                                            fig=visualise_samples(val_recon, scale=True),
                                            step=self.steps
                                        )
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_valid_error",
                                            fig=visualise_samples(val_input - val_recon, scale=False),
                                            step=self.steps
                                        )
                                    except KeyError:
                                        pass

                                # Sample and save validation if condition is passed
                                if valid_sampling_frequency and condition is not None and self.args.name != "autoencoder":
                                    try:
                                        val_sample, _ = self.sample_step(
                                            torch.randn_like(val_input),
                                            condition = val_condition,
                                        )
                                        
                                        self.save_figure(val_input, "validation", "sample_input")
                                        self.save_figure(val_sample, "validation", "sample_recon")
                                        self.save_figure(val_input - val_sample, "validation", "sample_error", scale=False)

                                        if logger:
                                            logger.log_figure(
                                                tag=f"{self.args.name.lower()}_valid_sample_input",
                                                fig=visualise_samples(val_input, scale=True),
                                                step=self.steps
                                            )
                                            logger.log_figure(
                                                tag=f"{self.args.name.lower()}_valid_sample_recon",
                                                fig=visualise_samples(val_sample, scale=True),
                                                step=self.steps
                                            )
                                            logger.log_figure(
                                                tag=f"{self.args.name.lower()}_valid_sample_error",
                                                fig=visualise_samples(val_input - val_sample, scale=False),
                                                step=self.steps
                                            )

                                    except NotImplementedError:
                                        pass

                                # Memory diagnostics
                                # gpu_diagnostics()

                        # Save training checkpoints
                        if (self.args.training.ckpt_freq > 0 and self.steps % self.args.training.ckpt_freq == 0) or epoch == self.args.training.n_epochs:
                            if not self.args.training.ckpt_last_only:
                                ckpt_name = f"{self.args.name.lower()}_ckpt_epoch_{self.epoch}_step_{self.steps}.pth"
                                ckpt_path = os.path.join(self.args.ckpt_path, ckpt_name)
                                self.save_checkpoint(ckpt_path)

                            # Update latest checkpoint
                            self.save_checkpoint()

                        self.steps += 1
                        # self.global_step_actor.increment.remote()
                        # print(">>>>>>>>>>>>>>>>>>>", self.global_step_actor.step)

                    self.epoch += 1

            except KeyboardInterrupt:
                self.save_checkpoint()
                # ray.stop()
                raise (KeyboardInterrupt)
        self.completed = True
        # ray.stop()
        return None

    @torch.no_grad()
    def validate(self) -> None:
        if self.valid_loader:
            total_loss = 0.
            for i, batch in enumerate(self.valid_loader):
                # Unpack batch into input and condition (if returned) and send to device
                if isinstance(batch, list):
                    input, condition = batch
                    # input, condition = input.float().to(self.device), condition.float().to(self.device)
                    input, condition = input.float(), condition.float()
                else:
                    input, condition = batch, None
                    # input = input.float().to(self.device)
                    input = input.float()

                 # Validation step   
                output = self.valid_step(input, condition=condition)
                try:
                    loss = output["loss"]
                except KeyError as e:
                    self.sys_logger.critical("Validation step output must be a dictionary containing the key 'loss'")
                    raise KeyError(e)
                total_loss += loss * input.shape[0]

            # Average validation losses
            avg_loss = total_loss / len(self.valid_loader.dataset)
            self.sys_logger.info(f"{self.args.name.lower().capitalize()} Average validation loss: {avg_loss.item()}")
        return None

    @torch.no_grad()
    def sample(self) -> None:
        start_time = time.time()
        if self.sample_loader:
            for i, sample_batch in enumerate(self.sample_loader):      
                # Unpack sampling batch into input and condition (if returned) and send to device.          
                if isinstance(sample_batch, list):
                    input, condition = sample_batch
                    # input, condition = input.float().to(self.device), condition.float().to(self.device)
                    input, condition = input.float(), condition.float()
                else:
                    input, condition = sample_batch, None
                    # input = input.float().to(self.device)
                    input = input.float()

                sample, _ = self.sample_step(input, condition=condition)
                self.save_figure(sample, "", "sample", save_tensor=True)

                self.sys_logger.info(f"{self.args.name.lower().capitalize()} Sampling batch {self.steps} / {len(self.sample_loader)} concluded. {self.eta(start_time)}")

                if self.args.sampling_only:
                    self.steps += 1
        return None



@ray.remote
class GlobalStepActor:
    def __init__(self):
        self.step = 0
    def increment(self,):
        self.step += 1