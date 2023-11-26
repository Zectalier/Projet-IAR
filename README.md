# BBRL - ALGOS + REDQ and DroQ

## Description

This is the subject of a university research project in which we had to conduct a reproducibility study of a Reinforcement Learning article using the DroQ algorithm based on REDQ. Since the library only contained the SAC algorithm, we implemented REDQ and DroQ ourselves. This library is designed for education purposes, it is mainly used to perform some practical experiences with various RL algorithms. It facilitates using optuna for tuning hyper-parameters and using rliable and statistical tests for analyzing the results.

## Installation

git clone https://github.com/osigaud/bbrl_algos.git

cd bbrl_algos

pip install -e .

We suggest using your favorite python environment (conda, venv, ...) as some further installations might be necessary

## Usage

go to src/bbrl_algos, choose your algorithm and run python3 your_algorithm.py

## References

- [**DROPOUT Q-FUNCTIONS FOR DOUBLY EFFICIENT REINFORCEMENT LEARNING**](https://arxiv.org/pdf/2110.02034.pdf)  
  *Takuya Hiraoka, Takahisa Imagawa, Taisei Hashimoto, Takashi Onishi, Yoshimasa Tsuruoka*
