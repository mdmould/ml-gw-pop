# ml-gw-pop

This repo contains material for the workshop [Scientific Machine Learning for Gravitational Wave Astronomy](https://icerm.brown.edu/program/topical_workshop/tw-25-smlgwa), 2-6 June 2025.

The [slides.pdf](slides.pdf) introduce three applications of machine learning for population inference of gravitational-wave sources.

There is a Python notebook for each of these applications. You can run with [Binder](https://mybinder.org/) by clicking [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mdmould/ml-gw-pop/HEAD), which will open this entire repo in a Jupyter session with the environment pre-installed and data pre-downloaded so you can get going straight away; however, the code will be very slow on CPU as Binder is not intended for heavier compute. Or you can use [Google Colab](https://colab.research.google.com/) (and GPUs!) with the following buttons for each notebook:

- [flow-variational-inference.ipynb](flow-variational-inference.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mdmould/ml-gw-pop/blob/main/flow-variational-inference.ipynb)
- [neural-posterior-estimation.ipynb](neural-posterior-estimation.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mdmould/ml-gw-pop/blob/main/neural-posterior-estimation.ipynb)
- [simulation-based-priors.ipynb](simulation-based-priors.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mdmould/ml-gw-pop/blob/main/simulation-based-priors.ipynb)

This will require installing packages and downloading data first, but should be much faster overall if you select a GPU runtime.

Alternatively, you can clone the repo and run locally or on a cluster - it is entirely self contained.

This material is focused on normalizing flows, which use neural networks for statistical inference. Below is a bonus notebook for Hamiltonian Monte Carlo:
- [hamiltonian-monte-carlo.ipynb](hamiltonian-monte-carlo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mdmould/ml-gw-pop/blob/main/hamiltonian-monte-carlo.ipynb)
