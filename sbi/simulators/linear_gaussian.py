import torch
from sbi.simulators.simulator import Simulator


class LinearGaussianSimulator(Simulator):
    """
    Implemenation of linear Gaussian simulator.
    Observations are generating by adding diagonal Gaussian noise of a specified variance
    to given parameters.
    """

    def __init__(self, dim=2, std=1, use_zero_ground_truth=True):
        """Set up linear Gaussian simulator.

        Keyword Arguments:
            dim {int} -- Dimension of the parameters and observations. (default: {2})
            std {int} -- Standard deviation of diagonal Gaussian shared across dimensions. (default: {1})
            use_zero_ground_truth {bool} -- Use the zero vector as a ground truth observation. (default: {True})
        """
        super().__init__()
        self._std = std
        self._dim = dim
        self._use_zero_ground_truth = use_zero_ground_truth

        # Generate ground truth samples to return if requested.
        self._ground_truth_samples = self._sample_ground_truth_posterior(
            num_samples=10000
        )

    def simulate(self, parameters):
        """Generate noisy observations of the given batch of parameters.
        
        Arguments:
            parameters {torch.Tensor} -- Batch of parameters.
        
        Returns:
            torch.Tensor -- Parameters plus diagonal Gaussian noise with shared variance across dimensions.
        """
        if parameters.ndim == 1:
            parameters = parameters[None, :]
        return parameters + self._std * torch.randn_like(parameters)

    def get_ground_truth_parameters(self):
        """True parameters always the zero vector.
        
        Returns:
            torch.Tensor -- Ground truth parameters.
        """
        return torch.zeros(self._dim)

    def get_ground_truth_observation(self):
        """Ground truth observation is either the zero vector, or a noisy observation of the
        zero vector.
        
        Returns:
            torch.Tensor -- Ground truth observation.
        """
        if self._use_zero_ground_truth:
            return torch.zeros(self._dim)
        else:
            return self._std * torch.randn(self._dim)

    def _sample_ground_truth_posterior(self, num_samples=1000):
        """Sample from ground truth posterior assuming prior is standard normal.
        
        Keyword Arguments:
            num_samples {int} -- Number of samples to draw. (default: {1000})
        
        Returns:
            torch.Tensor [num_samples, observation_dim] -- Batch of posterior samples.
        """
        mean = self.get_ground_truth_observation()
        std = torch.sqrt(torch.Tensor([self._std ** 2 / (self._std ** 2 + 1)]))
        c = torch.Tensor([1 / (self._std ** 2 + 1)])
        return c * mean + std * torch.randn(num_samples, self._dim)

    def get_ground_truth_posterior_samples(self, num_samples=1000):
        """Return first num_samples samples we have stored if there are enough,
        otherwise generate sufficiently many and returns those.

        Keyword Arguments:
            num_samples {int} -- Number of samples to generate. (default: {1000})
        
        Returns:
            torch.Tensor [batch_size, observation_dim] -- Batch of posterior samples.
        """
        if num_samples < self._ground_truth_samples.shape[0]:
            return self._ground_truth_samples[:num_samples]
        else:
            self._ground_truth_samples = self._sample_ground_truth_posterior(
                num_samples=num_samples
            )
            return self._ground_truth_samples

    @property
    def parameter_dim(self):
        """Number of dimensions of parameter vector.
        """
        return self._dim

    @property
    def observation_dim(self):
        """Number of dimensions data vector.
        """
        return self._dim

    @property
    def name(self):
        return "linear-gaussian"

    @property
    def normalization_parameters(self):
        """Mean and std for normalizing simulated data.
        """
        mean = torch.zeros(self._dim)
        std = torch.ones(self._dim)
        return mean, std
