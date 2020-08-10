# Can the algorithms deal with invalid data, e.g. NaN or inf?

Yes. By default, whenever a simulation returns at least one `NaN` or `inf`, it is 
completely excluded from the training data. In other words, the simulation is simply 
discarded.

In cases where a very large fraction of simulations return `NaN` or `inf`, discarding 
many simulations can be wasteful. Hence, it can be 
beneficial to manually substitute the 'invalid' values with a reasonable replacement. 
I.e., at the end of your simulation code, you search for invalid entries and replace 
them with a floating point number. Importantly, in order for neural network training 
work well, the floating point number should still be in a reasonable range, i.e. maybe a
 few standard deviations outside of 'good' values.

If you are running **multi-round** SNPE, however, things can go fully wrong if invalid
 data are encountered. In that case, you will get the following warning
```
When invalid simulations are excluded, multi-round SNPE-C can leak into the regions where parameters led to invalid simulations. This can lead to poor results.
```
Hence, if you are running multi-round SNPE and a significant fraction of simulations
returns at least one invalid number, we strongly recommend to manually replace the value
 in your simulation code as described above (or resort to single-round SNPE or use a 
 different method).