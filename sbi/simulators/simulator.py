class Simulator:
    """
    Base class for all simulator models.
    """

    def __init__(self):
        self.num_total_simulations = 0

    def __call__(self, parameters):
        """Simualte a batch of parameters. 
        
        Core method which returns a torch.Tensor batch of observations given a
        torch.Tensor batch of parameters. 

        Set up to be called as function because we want the simulator to be function at some point.  

        Arguments:
            parameters {torch.Tensor} -- batch of parameters.
        
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    @property
    def parameter_dim(self):
        """
        Dimension of parameters for simulator.

        :return: int parameter_dim
        """
        raise NotImplementedError

    @property
    def observation_dim(self):
        """
        Dimension of observations for simulator.
        TODO: decide whether observation_dim always corresponds to dimension of summary.

        :return: int observation_dim
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Name of the simulator.

        :return: str name
        """
        raise NotImplementedError
