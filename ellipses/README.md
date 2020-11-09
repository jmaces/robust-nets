# `ellipses`

Image reconstruction of 2D phantom ellipses from subsampled Fourier or Radon measurements.

## How to run this experiment

1. Check (and modify if necessary) the configuration file `config.py`. It specifies the directory paths for the data and results, as well as all relevant parameters of the experimental setup. By default, the data is stored in the subdirectory `raw_data` and results and model weights are stored in the subdirectory `results`.
2. Generate and prepare the data by running `data_management.py`.
3. Train networks using the scripts named `script_train_fourier*.py` or `script_train_radon_*.py` for the respective experiments with Fourier or Radon measurements.
4. Determine the regularization parameters of the total variation minimization reconstruction method by running `script_grid_search_l1_fourier.py` or `script_grid_search_l1_radon.py`.  
 *Remark: These scripts are designed to be run in parallel for multiple noise levels, making use of the batch-job capabilities of Sun Grid Engine cluster computing. You can adapt this to run sequentially, but be aware that this will be slow. *  
 Collect the results of the grid search by calling the `combine_results()` function from the respective grid search script.
5. Check (and modify if necessary) the configuration files `config_robustness_fourier.py` or `config_robustness_radon.py`. They specify the relevant parameters for the respective robustness analysis. Adapt the list of networks to compare according to the ones that you have actually trained.
6. Analyze different aspects of the robustness comparisons by running the scripts named `script_robustness_fourier*.py` or `script_robustness_radon*.py`.
