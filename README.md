# Ensemble Kalman Inversion for Optimal Experimental Design

This repository contains code to demonstrate the use of ensemble Kalman inversion as part of Bayesian optimal experimental design (OED) for inverse problems arising in geophysics.

## TODO:
### OED
Verification: 
 - For different numbers of sensors (e.g., 2, 4, 6, 10) generate a number (e.g. 10) of random designs. 
 - Generate a set of "test cases" by sampling some new sets of parameters from the prior and generating data. 
 - For each random design and the true design, characterise the posterior using each of the random designs and the greedy design. Use the same initial ensembles for each design. Might be a good idea to use a larger ensemble?
 - Compute the expectation of the trace of the posterior covariance for each design, and also the expectation of the different between the posterior mean and the truth.
 - Plot.