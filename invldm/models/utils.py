import os
import inspect
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.utils import scale2range

from .discriminator import NLayerDiscriminator
from .diffusion_model import DiffusionWrapper
from .autoencoder_model import AutoencoderWrapper, GaussianDistribution
from .losses import _divergence_fn, _perceptual_fn, _reconstruction_fn, _adversarial_fn


def _instance_autoencoder_model(args, device="cpu"):
    return AutoencoderWrapper(args, device)


def _instance_diffusion_model(autoencoder, args, device="cpu"):
    return DiffusionWrapper(autoencoder, args, device)

def _instance_discriminator_model(args, device="cpu"):
    return NLayerDiscriminator(args.model.out_channels,
                               args.params.disc_feature_channels,
                               args.params.disc_n_layers)#.to(device)


def _instance_optimiser(args, model):
    # Get optim class
    cls = getattr(optim, args.optim.optimiser)

    # Select only valid kwargs
    valid_kwargs = inspect.getfullargspec(cls).args
    kwargs = {}
    for vk in valid_kwargs:
        try:
            kwargs[vk] = args.optim.__dict__[vk]
        except KeyError:
            pass

    # change weight decay of biases to zero if prompted
    if not args.optim.bias_weight_decay:
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue           
            elif len(param.shape) == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else: decay_params.append(param)
        params = [{'params': no_decay_params, 'weight_decay': 0.}, {'params': decay_params, 'weight_decay': kwargs.pop("weight_decay")}]        
    else:
        params=model.parameters()

    # instance optimiser
    optimiser = cls(params, **kwargs)
    return optimiser


def _instance_lr_scheduler(args, optimiser):
    if args.optim.lr_scheduler.scheduler:
        cls = getattr(optim.lr_scheduler,
                      args.optim.lr_scheduler.scheduler)

        # Select only valid kwargs
        valid_kwargs = inspect.getfullargspec(cls).args
        kwargs = {}
        for vk in valid_kwargs:
            try:
                kwargs[vk] = args.optim.lr_scheduler.__dict__[vk]
            except KeyError:
                pass

        scheduler = cls(optimiser, **kwargs)
        return scheduler
    else:
        return None


def _instance_autoencoder_loss_fn(args):
    # Get correct loss functions from args
    div_fn = _divergence_fn(args)
    recon_fn = _reconstruction_fn(args)
    percep_fn = _perceptual_fn(args)


    # Function to combine all losses with their respective weights 
    def autoencoder_loss_fn(input: torch.Tensor,
                            recon: torch.Tensor,
                            mean: torch.Tensor,
                            log_var: torch.Tensor) -> torch.Tensor:
        
        # Evaluate reconstruction los function
        loss_recon = recon_fn(input, recon).mean()

        # Evaluate divergence los function
        loss_div = div_fn(mean, log_var).mean()

        # Evaluate perceptual loss -- LPIPS loss requires inputs in range [-1, 1]
        if args.model.perceptual_loss == "lpips":
            input, recon = scale2range(input, [-1, 1]), scale2range(recon, [-1, 1])
        loss_percep = percep_fn.to(input.device)(input, recon).mean()

        # Combine losses
        loss = args.params.recon_weight * loss_recon + \
            args.params.div_weight * loss_div + \
            args.params.perceptual_weight * loss_percep
        return loss
    
    return autoencoder_loss_fn


def _instance_discriminator_loss_fn(args):
    loss_fn = _adversarial_fn(args)

    def discriminator_loss_fn(prediction, is_real, apply_weight=False):
        target = torch.tensor(1.) if is_real else torch.tensor(0.)
        target = target.expand_as(prediction).to(prediction.device)
        loss = loss_fn(prediction, target)
        if apply_weight:
            loss = args.params.adversarial_weight * loss
        return loss

    return discriminator_loss_fn


def _instance_diffusion_loss_fn(args):
    diffusion_loss_fn = _reconstruction_fn(args)    
    return diffusion_loss_fn


def data_parallel_wrapper(module, device, **kwargs): 
    """
    Ensures code consistency for DataParallel when cpu is used.
    """
    if str(device) == "cpu":
        return DataParallelCPU(module)
    else:
        return nn.DataParallel(module, **kwargs)
    

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class DataParallelCPU(nn.Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module
        self.device_ids = []

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
