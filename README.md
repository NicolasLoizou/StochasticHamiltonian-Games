# Stochastic Hamiltonian Methods for Smooth Games
Code to reproduce the experiments reported in the paper:

```
@article{loizou2020stochastic}
  title={Stochastic Hamiltonian Methods for Smooth Games},
  author={Loizou, Nicolas and Berard, Hugo and Jolicoeur-Martineau, Alexia and Vincent, Pascal and Lacoste-Julien, Simon and Mitliagkas, Ioannis},
  journal={International Conference on Machine Learning},
  year={2020}
}
```

## Requirements
We provide a file `requirements.yml` with the list of requirements. To create a new conda environement with the correct requirements, run:
`conda env create -f requirements.yml`


## Bilinear Experiment
To reproduce the results of the bilinear experiment run:
`python run_bilinear.py [OUTPUT_PATH]`

**Note**:
- This will take around 4 hours to complete.
- You can get faster results by decreasing the number of iterations with the option `--num-iter [50000]` 
- You can also decrease the number of seeds with the option: `--num-seeds [5]` (using only 1 seed will reduce the amount of time it takes to complete by 5)
- You can also choose evaluate the different methods separately with the option: `-m [MODE]` where `[MODE]` can be (`"shgd-constant"`, `"shgd-decreasing"`, `"shgd-biased`, `"svrh"`, `"svre"`)


## Sufficiently-bilinear Experiment
To reproduce the results of the sufficiently-bilinear experiment run:
`python run_sufficiently_bilinear.py [OUTPUT_PATH]`

**Note**:
- This will take around 1 day to complete.
- You can get faster results by decreasing the number of iterations with the option `--num-iter [200000]` 
- You can also decrease the number of seeds with the option: `--num-seeds [5]` (using only 1 seed will reduce the amount of time it takes to complete by 5)
- You can also choose evaluate the different methods separately with the option: `-m [MODE]` where `[MODE]` can be (`"shgd-constant"`, `"shgd-decreasing"`, `"shgd-biased`, `"svrh"`, `"svrh-restart`, `"svre"`)

**To replicate the GAN experiments**
  * Run the jupyter notebook notebok/GAN.ipynb (This can be done by uploading GAN.ipynb to google Colab or making a copy of the read-only version here: https://colab.research.google.com/drive/16zh-Ma-vayaoHP6NhuWr6uJjoabDMsVb?usp=sharing)
