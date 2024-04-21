import torch
import torch.nn.functional as F

import warnings

from functools import partial

from ..utils.utils import scale2range

def null_fn(*args, **kwargs):
    return torch.tensor([0.])


def kl_div(mean, log_var):
    return 0.5 * torch.sum(torch.pow(mean, 2)
                            + log_var.exp() - 1.0
                            - log_var, dim=[1, 2, 3])


def laplacian2D(mesh, alpha=-0.2, beta=1.5):
    """ Helper function for AWLoss """
    xx, yy = mesh[:,:,0], mesh[:,:,1]
    x = torch.sqrt(xx**2 + yy**2) 
    T = 1 - torch.exp(-torch.abs(x) ** alpha) ** beta
    T = scale2range(T, [0.05, 1.])
    return T


def _divergence_fn(args):
    if not args.model.div_loss or args.params.div_weight<=0:
        return null_fn
    
    elif args.model.div_loss.lower() == "kl":
        return kl_div
    
    else:
        raise NotImplementedError(
            "Currently only supporting the KL divergence loss"
        )


def _perceptual_fn(args):
    if not args.model.perceptual_loss or args.params.perceptual_weight<=0:
        return null_fn
    
    elif args.model.perceptual_loss == "wiener":
        from awloss import AWLoss
        epsilon = args.params.__dict__.get("wiener_epsilon", 250.)
        alpha = args.params.__dict__.get("wiener_alpha", -0.2)
        beta = args.params.__dict__.get("wiener_beta", 1.5)
        filter_scale = args.params.__dict__.get("wiener_filter_scale", 2)

        awloss = AWLoss(filter_dim=2, method="fft", reduction="mean", store_filters=False,
                        epsilon=epsilon, filter_scale=filter_scale,
                        penalty_function=partial(laplacian2D, alpha=alpha, beta=beta))
        return awloss

    elif args.model.perceptual_loss == "lpips":
        import lpips
        lpips_model = args.params.__dict__.get("lpips_model", "alex")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lpips_fn = lpips.LPIPS(net=lpips_model, eval_mode=False, verbose=False)
        return lpips_fn

    else:
        raise NotImplementedError(
            f"Currently only supporting 'wiener' and 'lpips' as perceptual loss but got {args.model.perceptual_loss}"
        )


def _reconstruction_fn(args):
    loss_type = None
    try:
        loss_type = args.model.recon_loss.lower()
    except AttributeError:
        loss_type = args.model.loss.lower()

    if loss_type == "l2":
        return F.mse_loss
    elif loss_type == "l1":
        return F.l1_loss
    else:
        raise NotImplementedError(
            "Currently only supporting the L1 and L2 reconstruction losses"
        )


def _adversarial_fn(args):
    if not args.params.adversarial_mode or args.params.adversarial_weight<=0:
        return null_fn
    
    elif args.params.adversarial_mode == "vanilla":
        return F.mse_loss

    elif args.params.adversarial_mode == "lsgan":
        return F.binary_cross_entropy_with_logits

    else:
        raise NotImplementedError(
            f"Currently only supporting 'vanilla' and 'lsgan' adversarial modes but got {args.params.adversarial_mode}."
        )

