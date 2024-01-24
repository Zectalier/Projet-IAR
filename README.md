This project is a SB3 implementation of the DroQ and REDQ algorithm(described in https://arxiv.org/pdf/2110.02034.pdf and https://arxiv.org/pdf/2101.05982.pdf)

To run our code, use :
```
rl_algos.py [-h] -a {DroQ,SAC,REDQ} [-s STEPS] [-lr LEARNING_RATE] [-d DROPOUT] [-env ENVIRONMENT] [-c N_CRITICS] [-g GRADIENT_STEPS]
```

For example, if you want to run DroQ on the Hopper environement:
```
rl_algos.py -a DroQ -lr 3e-2 -d 0.01 -env Hopper-v4 -c 2 -s 1_000_000
```
