## FAQ

My simulator can process batches of parameters, can *sbi* harness this? 
- if your simulator is able to handle batches of `K` parameters, i.e. can take an input of shape `(K,N)` and then gives an output `(K,M)`, you need to set the `simulation_batch_size=K`.


   