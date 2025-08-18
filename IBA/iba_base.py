import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from utils.iba_utils import ifnone, get_module_by_name

from IBA.estimator import TorchWelfordEstimator

from utils.get_logger import logger
from tqdm import tqdm

class _InterruptExecution(Exception):
    pass


class _IBAForwardHook:
    def __init__(self, iba, idx):
        self.iba = iba
        self.idx = idx

    def __call__(self, m, inputs, outputs):
        return self.iba(outputs, self.idx)


class BaseIBA(nn.Module):
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
                 singular=False,
                 st=0,
                 ):
        super().__init__()
        
        self.relu = relu
        self.beta = beta
        self.min_std = min_std
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.alpha = None  # Initialized on first forward pass
        self.singular = singular
        self.sigmoid = nn.Sigmoid()
        self._buffer_capacity = None  # Filled on forward pass, used for loss
        self.sigma = sigma
        self.estimator = ifnone(estimator, TorchWelfordEstimator())
        self.device = device
        self._estimate = False
        self._active_neurons = None
        self._active_neurons_threshold = active_neurons_threshold
        self._restrict_flow = False
        self._interrupt_execution = False
        self._hook_handle = None
        self.model_name = model_name
        
        self.is_swin = 'swin' in model_name
        
        self.features = []
        self.original_features = []
        self._store_original = False
        self.st = st
            
        self.target_layer = None
        self.layer_estimator = None


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
    
    def _reset_alpha(self):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(self.initial_alpha)

    def _build(self):
        pass
    
    def clear(self):
        self.detach()
    
    def detach(self):
        """ Remove the bottleneck to restore the original model """
        if self._hook_handle is not None:
            if len(self._hook_handle) != 0:
                for h in self._hook_handle: 
                    h.remove()
            self._hook_handle = None
        # else:
        #     raise ValueError("Cannot detach hock. Either you never attached or already detached.")


    def forward(self, x, l):
        """
        You don't need to call this method manually.

        The IBA acts as a model layer, passing the information in `x` along to the next layer
        either as-is or by restricting the flow of infomration.
        We use it also to estimate the distribution of `x` passing through the layer.
        """
        if self._restrict_flow:
            return self._do_restrict_information(x, l)
        if self._estimate:
            self.layer_estimator[l](x)
        if self._interrupt_execution:
            raise _InterruptExecution()
        return x


    @staticmethod
    def _kl_div(r, lambda_, mean_r, std_r, normed=False):
        pass

        
    def _do_restrict_information(self, x, l_idx, alpha=None, interval=1):
        pass
    
    @contextmanager
    def enable_estimation(self):
        """
        Context manager to enable estimation of the mean and standard derivation.
        We recommend to use the `self.estimate` method.
        """
        self._estimate = True
        try:
            yield
        finally:
            self._estimate = False

    def reset_estimate(self):
        """
        Resets the estimator. Useful if the distribution changes. Which can happen if you
        trained the model more.
        """
        self.estimator = TorchWelfordEstimator()
        self.layer_estimator = dict()
        for i, target in enumerate(self.target_layer):
            self.layer_estimator[target] = TorchWelfordEstimator()
            
    def estimate(self, model, dataloader, device=None, n_samples=10000, reset=True):
        """ Estimate mean and variance using the welford estimator.
            Usually, using 10.000 i.i.d. samples gives decent estimates.

            Args:
                model: the model containing the bottleneck layer
                dataloader: yielding ``batch``'s where the first sample
                    ``batch[0]`` is the image batch.
                device: images will be transfered to the device. If ``None``, it uses the device
                    of the first model parameter.
                n_samples (int): run the estimate on that many samples
                reset (bool): reset the current estimate of the mean and std

        """
        # try:
        
        bar = tqdm(dataloader)
        
        if device is None:
            device = next(iter(model.parameters())).device
        if reset:
            self.reset_estimate()
        for batch in tqdm(dataloader):
            if self.layer_estimator[next(iter(self.layer_estimator.keys()))].n_samples() > n_samples:
                break
            with torch.no_grad(), self.enable_estimation():
                imgs = batch[0]
                model(imgs.to(device))
            if bar:
                bar.update(1)
                bar.set_description(f'{self.layer_estimator[next(iter(self.layer_estimator.keys()))].n_samples()}')
        if bar:
            bar.close()

        # After estimaton, feature map dimensions are known and
        # we can initialize alpha and the smoothing kernel
        if self.alpha is None:
            self._build()

    @contextmanager
    def restrict_flow(self):
        """
        Context mananger to enable information supression.

        Example:
            To make a prediction, with the information flow being supressed.::

                with iba.restrict_flow():
                    # now noise is added
                    model(x)
        """
        self._restrict_flow = True
        try:
            yield
        finally:
            self._restrict_flow = False
 
    def _iter_init(self):
        pass

    def analyze(self, input_t, class_idx, model_loss_fn, att_mask=None, mode="saliency",
                beta=None, optimization_steps=None, min_std=None, 
                lr=None, batch_size=None, active_neurons_threshold=0.01):
        pass
    
    def capacity(self):
        """
        Returns a tensor with the capacity from the last input, averaged
        over the redundant batch dimension.
        Shape is ``(self.channels, self.height, self.width)``
        """
        return self._buffer_capacity.mean(dim=0)
