# XXX: this class is deprecated and used only by NonLinearGaussian class
# XXX: we should rewrite nonlinear Gaussian and remove this class because
# XXX: we wanted to treat simulators as functions alltogether.
class Simulator:
    """
    Base class for all simulator models.
    """

    def __init__(self):
        self.num_total_simulations = 0

    def __call__(self, theta):
        """Simulate a batch of parameters theta.
        
        Core method which returns a Tensor batch of observations given a
        torch.Tensor batch of parameters theta.

        Set up to be called as function because we want the simulator to be function at
         some point.

        Args:
            theta: batch of parameters.
        
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    @property
    def parameter_dim(self):
        """
        Dimension of a single parameter set theta for simulator.

        :return: int theta_dim
        """
        raise NotImplementedError

    @property
    def observation_dim(self):
        """
        Dimension of a single simulation output x.
        TODO: decide whether observation_dim always corresponds to dimension of summary.

        :return: int x_dim
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Name of the simulator.

        :return: str name
        """
        raise NotImplementedError
