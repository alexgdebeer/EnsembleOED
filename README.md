# Ensemble Kalman Inversion for Optimal Experimental Design

This repository contains code to demonstrate the use of ensemble Kalman inversion as part of Bayesian optimal experimental design (OED) for inverse problems arising in geophysics.

## TODO:
### OED
 - For each candidate measurement location, condition on each of the N observations using EKI and record the trace of the posterior covariance matrix. This will give the best location.
 - Then condition of each pair of observations, under the constraint that the previously chosen observations must be part of the pair.
 - Repeat M times to obtain the final set of M measurements. 
 - Trial against other (random?) designs. Should I use the same observations as before or generate new ones? 