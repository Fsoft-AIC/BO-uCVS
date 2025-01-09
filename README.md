
<div align="center">

# [AAAI 2025] Bayesian Optimization for Unknown Cost-Varying Variable Subsets With No-Regret Costs

</div>

## Installation
The easiest way to install the required dependencies is to use Anaconda
on Linux (code was tested on Ubuntu 20.04.4 LTS). In this directory,
run
```bash
conda env create -f environment.yml
```
The environment can then be used with
```bash
conda activate uCVS
```
Alternatively, the dependencies can be installed manually using 
``environment.yml`` as reference.

## Running experiments
To run the baselines, first generate the desired list of experiments
to run with
```bash
python job_creator.py
```
This will create a ``jobs`` directory with ``jobs.txt`` containing
a list of experimental settings. After this, run
```bash
python job_runner.py
```
``job_runner.py`` will run indefinitely until all jobs in ``jobs.txt``
are exhausted. Multiple ``job_runner.py`` can be run at the same time
to run experiments in parallel.

To run the proposed method
```bash
python main.py
```
