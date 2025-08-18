import torch
import torch.nn as nn

class Estimator:

    def get_layer(self):
        raise NotImplementedError

    def shape(self):
        """ Get the shape of mean and std tensors """
        raise NotImplementedError

    def mean(self):
        """ Get accumulated mean per cell """
        raise NotImplementedError

    def std(self, stabilize=True):
        """ Get accumulated standard deviation per cell """
        raise NotImplementedError

    def p_zero(self):
        """ Get ratio of activation equal to zero per cell """
        raise NotImplementedError



class TorchWelfordEstimator(nn.Module):
    """
    Estimates the mean and standard derivation.
    For the algorithm see ``https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance``.

    Example:
        Given a batch of images ``imgs`` with shape ``(10, 3, 64, 64)``, the mean and std could
        be estimated as follows::

            # exemplary data source: 5 batches of size 10, filled with random data
            batch_generator = (torch.randn(10, 3, 64, 64) for _ in range(5))

            estim = WelfordEstimator(3, 64, 64)
            for batch in batch_generator:
                estim(batch)

            # returns the estimated mean
            estim.mean()

            # returns the estimated std
            estim.std()

            # returns the number of samples, here 10
            estim.n_samples()

            # returns a mask with active neurons
            estim.active_neurons()
    """
    def __init__(self):
        super().__init__()
        self.device = None  # Defined on first forward pass
        self.shape = None  # Defined on first forward pass
        self.register_buffer('_n_samples', torch.tensor([0], dtype=torch.long))

    def _init(self, shape, device, max_length=None):
        self.device = device
        self.shape = shape
        self.register_buffer('m', torch.zeros(*shape))
        self.register_buffer('s', torch.zeros(*shape))
        self.register_buffer('_neuron_nonzero', torch.zeros(*shape, dtype=torch.long))
        self.to(device)

    def forward(self, x, max_length=None):
        x = x.detach()
        # print(x.shape)
        # x = x.view(x.size(0) * x.size(1), -1)
        # print(x.shape)
        # if len(x.shape) == 3:
        #     x = x.view(x.size(0) * x.size(1), -1).unsqueeze(1)
        """ Update estimates without altering x """
        if self.shape is None:
            # Initialize runnnig mean and std on first datapoint
            self._init(x.shape[1:], x.device, max_length)
            # self._init(x[0].unsqueeze(0).shape, x.device, max_length)
        for xi in x:
            self._neuron_nonzero += (xi != 0.).long()
            old_m = self.m.clone()
            self.m = self.m + (xi-self.m) / (self._n_samples.float() + 1)
            self.s = self.s + (xi-self.m) * (xi-old_m)
            self._n_samples += 1
        return x

    def n_samples(self):
        """ Returns the number of seen samples. """
        return int(self._n_samples.item())

    def mean(self):
        """ Returns the estimate of the mean. """
        return self.m

    def std(self):
        """returns the estimate of the standard derivation."""
        return torch.sqrt(self.s / (self._n_samples.float() - 1))

    def active_neurons(self, threshold=0.01):
        """
        Returns a mask of all active neurons.
        A neuron is considered active if ``n_nonzero / n_samples  > threshold``
        """
        return (self._neuron_nonzero.float() / self._n_samples.float()) > threshold

