# Solving Inverse Problems With Deep Neural Networks - Robustness Included?

[![GitHub license](https://img.shields.io/github/license/jmaces/robust-nets)](https://github.com/jmaces/robust-nets/blob/master/LICENSE)
[![code-style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-Pytorch-1f425f.svg)](https://pytorch.org/)

This repository provides the official implementation of the paper [Solving Inverse Problems With Deep Neural Networks - Robustness Included?](http://arxiv.org/abs/2011.04268) by M. Genzel, J. Macdonald, and M. März (2020).

## Content

This repository contains subfolders for five experimental scenarios. Each of them is independent of the others.

- [`tvsynth`](tvsynth) : Signal recovery of piecewise constant 1D signals (following a total variation synthesis model) from random Gaussian measurements.

- [`mnist`](mnist) : Signal recovery of approximately piecewise constant 1D signals (vectorized MNIST images) from random Gaussian measurements.

- [`ellipses`](ellipses) : Image reconstruction of 2D phantom ellipses from subsampled Fourier or Radon measurements.

- [`fastmri-radial`](fastmri-radial) : Image reconstruction from multi-coil MRI data with radial line subsampling mask.

- [`fastmri-challenge`](fastmri-challenge) : Image reconstruction from single-coil MRI data with subsampling mask according to the [fastMRI](https://fastmri.org/) challenge (2019).

## Requirements

The package versions are the ones we used. Other versions might work as well.

`matplotlib` *(v3.1.3)*  
`numpy` *(v1.18.1)*  
`pandas` *(v1.0.5)*  
`piq` *(v0.4.1)*  
`python` *(v3.8.3)*  
`pytorch` *(v1.4.0)*  
`scikit-image` *(v0.16.2)*  
`torchvision` *(v0.5.0)*  
`tqdm` *(v4.46.0)*

`h5py` *(v2.10.0)* (only for `fastmri-radial` and `fastmri-challenge`)  
`odl` *(v0.7.0)* (only for `ellipses`)  
`pytorch-radon` *(v0.1.3)* (only for `ellipses`)  
`torch-cg` *(v1.0.1)* (only for `ellipses`)  

## Usage

Each of the individual experiment subfolders contains configuration files as well
as scripts for preparing the data, for training the neural networks, for obtaining total variation minimization reconstructions, and for finding adversarial perturbations.

The details are described within in each subfolder.

## Acknowledgements

Our implementation of the U-Net is based on and adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch/.  
Our implementation of the Tiramisu network is based on and adapted from https://github.com/bfortuner/pytorch_tiramisu/.  
For the processing of the fastMRI data we relied on utilty code from https://github.com/facebookresearch/fastMRI.

Thank you for making your code available.

## License

This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
