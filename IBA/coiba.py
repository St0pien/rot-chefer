import torch
import torch.nn as nn
import numpy as np
import warnings
from utils.iba_utils import ifnone

from IBA.iba import IBA

import einops

from utils.get_logger import logger


class CoIBA(IBA):
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
                 optimization_steps=10,
                 lr=1,
                 batch_size=10,
                 estimator=None,
                 relu=False,
                 st=0,
                 ):
        super(CoIBA, self).__init__(
                    model_name=model_name,
                    device=device,
                    sigma=sigma,
                    beta=beta,
                    optimization_steps=optimization_steps,
                    lr=lr,
                    batch_size=batch_size,
                    estimator=estimator,
                    relu=relu,
                    st=st,
                    )


    def _build(self):
        """
        Initialize alpha with the same shape as the features.
        We use the estimator to obtain shape and device.
        mode 0: read-out bottleneck
        mode 1: per-sample bottleneck
        mode 2: per-sample bottleneck with DPI constraints
        """
        if self.layer_estimator[next(iter(self.layer_estimator.keys()))].n_samples() <= 0:
            raise RuntimeWarning("You need to estimate the feature distribution"
                                " before using the bottleneck.")
        shape = self.layer_estimator[next(iter(self.layer_estimator.keys()))].shape
        device = self.layer_estimator[next(iter(self.layer_estimator.keys()))].device
           
        if len(shape) == 2:
            self.alpha = nn.Parameter(torch.full((shape[0] - self.st, 1), self.initial_alpha, device=device),
                                    requires_grad=True)
        elif len(shape) == 3:
            if self.is_swin:
                self.alpha = nn.Parameter(torch.full((shape[0] - self.st, *shape[1:-1], 1), self.initial_alpha, device=device),
                                        requires_grad=True)
            else: # For CNN
                self.alpha = nn.Parameter(torch.full((1, *shape[1:]), self.initial_alpha, device=device),
                                        requires_grad=True)
                    
        logger.info(f'Alpha shape: {self.alpha.shape}')
        self.smooth = None
       
       
    def _do_restrict_information(self, x, l_idx, alpha=None, interval=1):
        """ Selectively remove information from x by applying noise """
        
        if len(x.size()) == 4 and self.is_swin:
            x = einops.rearrange(x, 'b h w c -> b c h w')
        
        if alpha is None:
            alpha = self.alpha

        if self.st != 0:
            x = x[:, self.st:, :, :] if len(x.size()) == 4 else x[:, self.st:, :]
        
        self.xs[l_idx] = x
        
        assert self.l_idx <= len(self.target_layer)
        
        if self.l_idx == len(self.target_layer):
            self._buffer_capacity = 0
            i = 0; t = self.target_layer[i]
                
            μ_r = self.layer_estimator[t].mean()[self.st:]
            σ_r = self.layer_estimator[t].std()[self.st:]
            σ_r = torch.max(σ_r, 0.01*torch.ones_like(σ_r))
            _kl_in_1 = self.xs[t]
        
            λ_m = self.sigmoid(alpha)
            if len(_kl_in_1.size()) == 4 and self.is_swin:
                μ_r = einops.rearrange(μ_r, 'h w c -> c h w')
                σ_r = einops.rearrange(σ_r, 'h w c -> c h w')
                λ_m = einops.rearrange(λ_m, 'h w c -> c h w')

            λ_m = λ_m.expand(_kl_in_1.shape[0], -1, -1, -1) if len(_kl_in_1.size()) == 4 else λ_m.expand(_kl_in_1.shape[0], -1, -1)
            
            _kl = self._kl_div(_kl_in_1, λ_m, μ_r, σ_r)
                            
            self._buffer_capacity += _kl 
        
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
        
        if len(x.size()) == 4 and self.is_swin:
            z = einops.rearrange(z, 'b c h w -> b h w c')
        
        return z
    
    
    def analyze(self, input_t, class_idx, model_loss_fn, 
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
        
        # To deal with cross-entropy bug
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