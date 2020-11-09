# `tvsynth`

Signal recovery of piecewise constant 1D signals (following a total variation synthesis model) from random Gaussian measurements.

## How to run this experiment

1. Check (and modify if necessary) the configuration file `config.py`. It specifies the directory paths for the data and results, as well as all relevant parameters of the experimental setup. By default, the data is stored in the subdirectory `raw_data` and results and model weights are stored in the subdirectory `results`.
2. Generate and prepare the data by running `data_management.py`.
3. Train networks using the scripts named `script_train_*.py`.
4. Check (and modify if necessary) the configuration file `config_robustness.py`. It specifies the relevant parameters for the robustness analysis. Adapt the list of networks to compare according to the ones that you have actually trained.
5. Analyze different aspects of the robustness comparisons by running the scripts named `script_robustness_*.py`.
