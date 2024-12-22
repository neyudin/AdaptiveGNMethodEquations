# Description
Code to conduct experiments for the paper [**Adaptive Gauss-Newton Method for Solving Systems of Nonlinear Equations**](https://rdcu.be/cGkrt).

## Overview

* *run_experiments.py* — main script to perform experiments;
* *oracles.py* — contains classes for optimization criteria;
* *opt_utils.py* — auxiliary functions for optimizers;
* *optimizers.py* — contains Gauss-Newton optimization algorithms;
* *benchmark_utils.py* — routines for designed experiments;
* *plotting.py* — routines for plotting results;
* *print_time.py* — routines for registering time measurements.

Print help in command line in repository directory to list all hyperparameters of the experiments:
```
    python run_experiments.py -h
```
Run the following command in command line in repository directory to obtain all experiment data in current directory:
```
    python run_experiments.py
```

## Requirements

* [NumPy](https://numpy.org/);
* [Matplotlib](https://matplotlib.org/);
* [Seaborn](https://seaborn.pydata.org/).

## References

<a id="1">[1]</a> Yudin, N.E. Adaptive Gauss–Newton Method for Solving Systems of Nonlinear Equations // Doklady Mathematics, 2021, vol. 104, no. 2, pp. 293-296, [doi: https://doi.org/10.1134/S1064562421050161](https://doi.org/10.1134/S1064562421050161).
