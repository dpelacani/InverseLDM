"""
[1] https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion/latent_diffusion.py


---
title: Latent Diffusion Models
summary: >
 Annotated PyTorch implementation/tutorial of latent diffusion models from paper
 High-Resolution Image Synthesis with Latent Diffusion Models
---

# Latent Diffusion Models

Latent diffusion models use an auto-encoder to map between image space and
latent space. The diffusion model works on the latent space, which makes it
a lot easier to train.
It is based on paper
[High-Resolution Image Synthesis with Latent Diffusion Models](https://papers.labml.ai/paper/2112.10752).

They use a pre-trained auto-encoder and train the diffusion U-Net on the latent
space of the pre-trained auto-encoder.

For a simpler diffusion implementation refer to our [DDPM implementation](../ddpm/index.html).
We use same notations for $\alpha_t$, $\beta_t$ schedules, etc.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .autoencoder_model import Autoencoder
from .unet import UNetModel
from .samplers import *

import argparse


class DiffusionWrapper(nn.Module):
    """
    *This is an empty wrapper class around the [U-Net](model/unet.html).
    We keep this to have the same model structure as
    [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
    so that we do not have to map the checkpoint weights explicitly*.
    """

    def __init__(self,
                 autoencoder: Autoencoder,
                 args: argparse.Namespace, 
                 device: str):
        super().__init__()
        self.args = args
        self.autoencoder = autoencoder

        self.unet = UNetModel(
            in_channels= autoencoder.emb_channels,
            out_channels= autoencoder.emb_channels,
            channels= args.model.feature_channels,
            n_res_blocks= args.model.num_res_blocks,
            attention_levels= args.model.attention,
            channel_multipliers= args.model.channels_mult,
            n_heads= args.model.num_attn_heads,
            tf_layers= args.model.num_transformer_layers,
            cond_in_channels = args.model.condition.in_channels,
            d_cond=args.model.condition.feature_channels)

        self.ldm = LatentDiffusion(
                 unet_model=self.unet,
                 autoencoder=self.autoencoder,
                 latent_scaling_factor=args.params.latent_scaling_factor,
                 n_steps=args.params.num_diffusion_timesteps,
                 linear_start=args.params.beta_start,
                 linear_end=args.params.beta_end)
        
        if args.params.sampler.lower() == "ddim":
            self.sampler = DDIMSampler(model=self.ldm,
                            n_steps=args.params.num_diffusion_timesteps,
                            ddim_discretize="uniform",
                            ddim_eta=0.)
        elif args.params.sampler.lower() == "ddpm":
            self.sampler = DDPMSampler(model=self.ldm)
        else:
            raise NotImplementedError(f"{args.params.sampler} sampler not implemented.")
    
    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.unet.parameters())).device

    def forward(self, x: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                noise: Optional[torch.Tensor] = None,
                **kwargs):
        
        # Get batch size
        batch_size = x.shape[0]

        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.ldm.n_steps, (batch_size,), device=x.device, dtype=torch.long)

        # Encode x to latent 
        z0 = self.ldm.autoencoder_encode(x, condition)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(z0)

        # Sample $x_t$ for $q(x_t|x_0)$
        zt = self.sampler.q_sample(z0, t, noise=noise)

        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        noise_pred = self.ldm(zt, t, condition)

        return noise, noise_pred


class LatentDiffusion(nn.Module):
    """
    ## Latent diffusion model

    """

    def __init__(self,
                 unet_model: UNetModel,
                 autoencoder: Autoencoder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                 ):
        """
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param autoencoder: is the [AutoEncoder](model/autoencoder.html)
        :param clip_embedder: is the [CLIP embeddings generator](model/clip_embedder.html)
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()
        # Wrap the [U-Net](model/unet.html) to keep the same model structure as
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
        self.unet_model = unet_model

        # Auto-encoder and scaling factor
        self.first_stage_model = autoencoder
        self.latent_scaling_factor = latent_scaling_factor


        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64, device=self.device) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # $\alpha_t = 1 - \beta_t$
        alpha = 1. - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.unet_model.parameters())).device

    def autoencoder_encode(self, image: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """
        ### Get scaled latent space representation of the image

        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        _, _ = self.first_stage_model.encode(image, condition)
        return self.latent_scaling_factor * self.first_stage_model.sample()

    def autoencoder_decode(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """
        ### Get image from the latent representation

        We scale down by the scaling factor and then decode.
        """
        return self.first_stage_model.decode(z / self.latent_scaling_factor, condition)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None):
        """
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning c $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.unet_model(x, t, c)
    

