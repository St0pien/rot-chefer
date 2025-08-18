import torch
import torch.nn as nn

import numpy as np

import warnings
from utils.iba_utils import ifnone, get_module_by_name

from IBA.iba_base import BaseIBA

import einops

from utils.get_logger import logger

class _IBAForwardHook:
    def __init__(self, iba, idx):
        self.iba = iba
        self.idx = idx

    def __call__(self, m, inputs, outputs):
        return self.iba(outputs, self.idx)


class IBA(BaseIBA):
    """
    IBA finds relevant features of your model by applying noise to
    intermediate features.

    Example: ::

        model = Net()
        # Create the Per-Sample Bottleneck:
        iba = IBA(model.conv4)

        # Estimate the mean and variance.
        iba.estimate(model, datagen)

        img, target = next(iter(datagen(batch_size=1)))

        # Closure that returns the loss for one batch
        model_loss_closure = lambda x: F.nll_loss(F.log_softmax(model(x), target)

        # Explain class target for the given image
        saliency_map = iba.analyze(img.to(dev), model_loss_closure)
        plot_saliency_map(img.to(dev))


    Args:
        layer: The layer after which to inject the bottleneck
        sigma: The standard deviation of the gaussian kernel to smooth
            the mask, or None for no smoothing
        beta: Weighting of model loss and mean information loss.
        min_std: Minimum std of the features
        lr: Optimizer learning rate. default: 1. As we are optimizing
            over very few iterations, a relatively high learning rate
            can be used compared to the training of the model itself.
        batch_size: Number of samples to use per iteration
        input_or_output: Select either ``"output"`` or ``"input"``.
        initial_alpha: Initial value for the parameter.
    """
    def __init__(self,
                 model_name=None,
                 device=None,
                 sigma=1.,
                 beta=10,
                 min_std=0.01,
                 optimization_steps=10,
                 lr=1,
                 batch_size=10,
                 initial_alpha=5.0,
                 active_neurons_threshold=0.01,
                 estimator=None,
                 relu=False,
                 st=0,
                 ):
        super().__init__(
            model_name=model_name,
            device=device,
            sigma=sigma,
            beta=beta,
            min_std=min_std,
            optimization_steps=optimization_steps,
            lr=lr,
            batch_size=batch_size,
            initial_alpha=initial_alpha,
            active_neurons_threshold=active_neurons_threshold,
            estimator=estimator,
            relu=relu,
            st=st
        )


    def iba_preset(self, model, target_layers):
        
        modules, layer_estimator = get_module_by_name(model, target_layers)
            
        if self.layer_estimator is None:
            self.layer_estimator = layer_estimator
        if self.target_layer is None:
            self.target_layer = target_layers
            logger.info(f'Target Layer: {self.target_layer}')
        self._hook_handle = []
        
        for m in modules:
            self._hook_handle.append(m[1].register_forward_hook(
                                        _IBAForwardHook(self, idx=m[0])))

        if len(self.target_layer) == 0:
            for m, _  in model.named_modules():
                logger.info(m)
            logger.error(f'(Fatal Error) {self.target_layer} is not in model')
            logger.error('System will be terminated')
            import sys; sys.exit()
        
        
        return self.target_layer
    
    
    def _build(self):
        """
        Initialize alpha with the same shape as the features.
        We use the estimator to obtain shape and device.
        """
        if self.layer_estimator[next(iter(self.layer_estimator.keys()))].n_samples() <= 0:
            raise RuntimeWarning("You need to estimate the feature distribution"
                                " before using the bottleneck.")
        shape = self.layer_estimator[next(iter(self.layer_estimator.keys()))].shape
        device = self.layer_estimator[next(iter(self.layer_estimator.keys()))].device
           
        if len(shape) == 2:
            self.alpha = nn.Parameter(torch.full((shape[0] - self.st, *shape[1:]), self.initial_alpha, device=device),
                                    requires_grad=True)
        elif len(shape) == 3:
            self.alpha = nn.Parameter(torch.full(shape, self.initial_alpha, device=device),
                                    requires_grad=True)
                    
        logger.info(f'Alpha shape: {self.alpha.shape}')
        self.smooth = None
    
    
    @staticmethod
    def _kl_div(r, lambda_, mean_r, std_r):
        """Computes the KL Divergence between the noise (Q(Z)) and
            the noised activations P(Z|R))."""
        # We want to compute: KL(P(Z|R) || Q(Z))
        # As both distributions are normal distribution, we need the
        # mean and variance of both. We can simplify the computation
        # by normalizing R to R' = (R - E[R]) / std(R) [1] and ε' ~ N(0,1).
        # The KL divergence is invariant under scaling.
        #
        # For the mean and var of Z|R, we have:
        # Z' = λ * R' + (1 - λ) * ε'
        # E[Z'|R'] = λ E[R']                    [2]
        # Var[Z'|R'] = (1 - λ)**2               [3]
        # Remember that λ(R) and therefore the λ becomes a constant if
        # R is given.
        #

        # Normalizing [1]
        r_norm = (r - mean_r) / std_r

        # Computing mean and var Z'|R' [2,3]
        var_z = (1 - lambda_) ** 2
        mu_z = r_norm * lambda_

        log_var_z = torch.log(var_z)

        # For computing the KL-divergence:
        # See eq. 7: https://arxiv.org/pdf/1606.05908.pdf
        capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
        return capacity


    def _do_restrict_information(self, x, l_idx, alpha=None, interval=1):
        
        """ Selectively remove information from x by applying noise """
        if len(x.size()) == 4 and self.is_swin:
            x = einops.rearrange(x, 'b h w c -> b c h w')
            
        if alpha is None:
            alpha = self.alpha
            
        # self.st = 1 means perturbation will not be applied to CLS token
        if self.st != 0:
            orig_x = x
            x = x[:, self.st:, :, :] if len(x.size()) == 4 else x[:, self.st:, :]
        
        # self._buffer_capacity = 0
        
        t = self.target_layer[0]
        
        assert self.l_idx <= len(self.target_layer) # For debugging
        
        μ_r = self.layer_estimator[t].mean()[self.st:]
        σ_r = self.layer_estimator[t].std()[self.st:]
        σ_r = torch.max(σ_r, 0.01*torch.ones_like(σ_r))
        
        λ_m = self.sigmoid(alpha)

        # To deal with Swin Transformer
        if len(x.size()) == 4 and self.is_swin:
            μ_r = einops.rearrange(μ_r, 'h w c -> c h w')
            σ_r = einops.rearrange(σ_r, 'h w c -> c h w')
            λ_m = einops.rearrange(λ_m, 'h w c -> c h w')

        λ_m = λ_m.expand(x.shape[0], -1, -1, -1) if len(x.size()) == 4 else λ_m.expand(x.shape[0], -1, -1)
        
        # Since the ViT has (P^2 + 1) number of tokens with class token, we omit the smooth function
        _kl = self._kl_div(x, λ_m, μ_r, σ_r)
        
        self._buffer_capacity = _kl 
        
        μ_curr = self.layer_estimator[l_idx].mean()[self.st:]
        σ_curr = self.layer_estimator[l_idx].std()[self.st:]
        σ_curr = torch.max(σ_curr, 0.01*torch.ones_like(σ_curr))
        
        λ_m = self.sigmoid(alpha)
        
        if len(x.size()) == 4 and self.is_swin:
            μ_curr = einops.rearrange(μ_curr, 'h w c -> c h w')
            σ_curr = einops.rearrange(σ_curr, 'h w c -> c h w')
            λ_m = einops.rearrange(λ_m, 'h w c -> c h w')  
        
        λ_m = λ_m.expand(x.shape[0], -1, -1, -1) if len(x.size()) == 4 else λ_m.expand(x.shape[0], -1, -1)
        
        eps = x.data.new(x.size()).normal_()
    
        ε1 = σ_curr * eps + μ_curr
        
        z = λ_m * x + (1 - λ_m) * ε1
        
        self.l_idx += 1
        
        # To deal with Swin Transformer
        if len(x.size()) == 4 and self.is_swin:
            z = einops.rearrange(z, 'b c h w -> b h w c')
        
        if self.st != 0:
            z = torch.cat((orig_x[:, :self.st, :, :], z), dim=1) if len(orig_x.size()) == 4 else torch.cat((orig_x[:, :self.st, :], z), dim=1)
        
        return z
    
    
    def _iter_init(self):
        self.xs = dict()
        self.l_idx = 1; # Layer index
        self._cosine = 0
        

    def analyze(self, input_t, class_idx, model_loss_fn, att_mask=None, mode="saliency",
                beta=None, optimization_steps=None, min_std=None, 
                lr=None, batch_size=None, active_neurons_threshold=0.01,
                is_imagenet_a=False, is_imagenet_r=False):
        """
        Generates a heatmap for a given sample. Make sure you estimated mean and variance of the
        input distribution.

        Args:
            input_t: input image of shape (1, C, H W)
            model_loss_fn: closure evaluating the model
            mode: how to post-process the resulting map: 'saliency' (default) or 'capacity'
            beta: if not None, overrides the bottleneck beta value
            optimization_steps: if not None, overrides the bottleneck optimization_steps value
            min_std: if not None, overrides the bottleneck min_std value
            lr: if not None, overrides the bottleneck lr value
            batch_size: if not None, overrides the bottleneck batch_size value
            active_neurons_threshold: used threshold to determine if a neuron is active

        Returns:
            The heatmap of the same shape as the ``input_t``.
        """
        assert input_t.shape[0] == 1, "We can only fit one sample a time"
        
        if self.alpha is None:
            self._build()
            
        if not isinstance(input_t, torch.Tensor):
            input_t = torch.tensor(input_t).cuda(self.device)
        if not isinstance(class_idx, torch.Tensor):
            if not isinstance(class_idx, tuple):
                class_idx = torch.tensor(class_idx).cuda(self.device)
            else:
                raise ValueError("class_idx should be a tensor or a tuple of tensors")
            
        # TODO: is None
        beta = ifnone(beta, self.beta)
        optimization_steps = ifnone(optimization_steps, self.optimization_steps)
        min_std = ifnone(min_std, self.min_std)
        lr = ifnone(lr, self.lr)
        batch_size = ifnone(batch_size, self.batch_size)
        active_neurons_threshold = ifnone(active_neurons_threshold, self._active_neurons_threshold)
        
        batch = input_t.expand(batch_size, -1, -1, -1)
        
        # Reset from previous run or modifications
        self._reset_alpha()
        
        optimizer = torch.optim.Adam(lr=lr, params=[self.alpha])

        if self.layer_estimator[next(iter(self.layer_estimator.keys()))].n_samples() < 1000:
            warnings.warn(f"Selected estimator was only fitted on {self.estimator.n_samples()} "
                        f"samples. Might not be enough! We recommend 10.000 samples.")

        opt_range = range(optimization_steps)
        target = torch.tensor([class_idx,] * batch_size).to(input_t.device) if input_t.size(0) == 1 else class_idx
        
        with self.restrict_flow():
            for _ in opt_range:
                self._iter_init()
                optimizer.zero_grad()

                model_loss = model_loss_fn(batch, target, is_imagenet_a=is_imagenet_a, is_imagenet_r=is_imagenet_r)
                
                # Taking the mean is equivalent of scaling the sum with 1/K
                information_loss = self.capacity().mean()
                loss = model_loss + beta * information_loss 
                
                loss.backward()
                optimizer.step()

                self.features = []
                
        return self._get_saliency(shape=input_t.shape[2:])
    
    
    def capacity(self):
        """
        Returns a tensor with the capacity from the last input, averaged
        over the redundant batch dimension.
        Shape is ``(self.channels, self.height, self.width)``
        """
        return self._buffer_capacity.mean(dim=0)
    

    def _get_saliency(self, mode='saliency', shape=None):
        capacity = self.capacity().detach()
        
        if mode == "saliency":
            # In bits, summed over channels, scaled to input
            # return to_saliency_map(capacity_np, shape)
            saliency_map = capacity / float(np.log(2))
            if shape is not None:
                ho, wo = saliency_map.shape
                h, w = shape
                # Scale bits to the pixels
                saliency_map *= (ho*wo) / (h*w)
                return saliency_map
            else:
                return saliency_map
                
        elif mode == "capacity":
            # In bits, not summed, not scaled
            return capacity / float(np.log(2))
        else:
            raise ValueError