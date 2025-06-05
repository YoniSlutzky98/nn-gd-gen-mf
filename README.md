# Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study
Official implementation for the experiments in [Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](https://arxiv.org/abs/2506.03931), based on the [PyTorch](https://pytorch.org/) library.

## Installing Requirements

Tested with Python 3.9. The ```requirements.txt``` file includes the required dependencies, which can be installed via:

```
pip install -r requirements.txt
```

## Experiments

The experiments were carried out using a single Nvidia RTX A6000 GPU with 48GB RAM. All experiments attempt to use a GPU if one is present and unused, and use CPU otherwise.
**Important** - It is highly suggested to verify your machine's RAM and adjust the batch size for G&C appropriately.

### Running Width Experiments 

The experiments train matrix factorization using G&C and GD with fixed depth and varying widths.
By default, results and figures are saved to subdirectories in the working directory. 
For a quick start, use the following command line script to run the experiment in Figure 1 for the linear activation function:   

```
python width_experiments.py --config ./configs/width/width_gt_rank=1_act=Linear_gnc_init=gauss.json
```

Various hyperparameters are configurable, see ```./common/parser.py```. 

The following table details which experiment is reproduced via which config file in ```./configs/width```:

| Experiment                      | Config                                                                   |
|---------------------------------|--------------------------------------------------------------------------|
| Figure 1, Linear Activation     | ```width_gt_rank=1_act=Linear_gnc_init=gauss.json```                     |
| Figure 1, Tanh Activation       | ```width_gt_rank=1_act=Tanh_gnc_init=gauss.json```                       |
| Figure 1, Leaky ReLU Activation | ```width_gt_rank=1_act=LeakyReLU_gnc_init=gauss.json```                  |
| Figure 3, Linear Activation     | ```width_gt_rank=1_act=Linear_gnc_init=gauss_gd_momentum=0.9.json```     |
| Figure 3, Tanh Activation       | ```width_gt_rank=1_act=Tanh_gnc_init=gauss_gd_momentum=0.9.json```       |
| Figure 3, Leaky ReLU Activation | ```width_gt_rank=1_act=LeakyReLU_gnc_init=gauss_gd_momentum=0.9.json```  |
| Figure 5, Linear Activation     | ```width_gt_rank=2_act=Linear_gnc_init=gauss.json```                     |
| Figure 5, Tanh Activation       | ```width_gt_rank=2_act=Tanh_gnc_init=gauss.json```                       |
| Figure 5, Leaky ReLU Activation | ```width_gt_rank=2_act=LeakyReLU_gnc_init=gauss.json```                  |
| Figure 7, Linear Activation     | ```width_gt_rank=1_act=Linear_gnc_init=unif.json```                      |
| Figure 7, Tanh Activation       | ```width_gt_rank=1_act=Tanh_gnc_init=unif.json```                        |
| Figure 7, Leaky ReLU Activation | ```width_gt_rank=1_act=LeakyReLU_gnc_init=unif.json```                   |
| Figure 9, Linear Activation     | ```width_gt_rank=1_act=Linear_gnc_init=gauss_completion.json```          |
| Figure 9, Tanh Activation       | ```width_gt_rank=1_act=Tanh_gnc_init=gauss_completion.json```            |
| Figure 9, Leaky ReLU Activation | ```width_gt_rank=1_act=LeakyReLU_gnc_init=gauss_completion.json```       |

### Running Depth Experiments 

The experiments train matrix factorization using G&C and GD with fixed width and varying depths.
By default, results and figures are saved to subdirectories in the working directory. 
For a quick start, use the following command line script to run the experiment in Figure 2 for the linear activation function:   

```
python width_experiments.py --config ./configs/depth/depth_gt_rank=1_act=Linear_gnc_init=gauss.json
```

Various hyperparameters are configurable, see ```./common/parser.py```. 

The following table details which experiment is reproduced via which config file in ```./configs/depth```:

| Experiment                       | Config                                                                   |
|----------------------------------|--------------------------------------------------------------------------|
| Figure 2, Linear Activation      | ```depth_gt_rank=1_act=Linear_gnc_init=gauss.json```                     |
| Figure 2, Tanh Activation        | ```depth_gt_rank=1_act=Tanh_gnc_init=gauss.json```                       |
| Figure 2, Leaky ReLU Activation  | ```depth_gt_rank=1_act=LeakyReLU_gnc_init=gauss.json```                  |
| Figure 4, Linear Activation      | ```depth_gt_rank=1_act=Linear_gnc_init=gauss_gd_momentum=0.9.json```     |
| Figure 4, Tanh Activation        | ```depth_gt_rank=1_act=Tanh_gnc_init=gauss_gd_momentum=0.9.json```       |
| Figure 4, Leaky ReLU Activation  | ```depth_gt_rank=1_act=LeakyReLU_gnc_init=gauss_gd_momentum=0.9.json```  |
| Figure 6, Linear Activation      | ```depth_gt_rank=2_act=Linear_gnc_init=gauss.json```                     |
| Figure 6, Tanh Activation        | ```depth_gt_rank=2_act=Tanh_gnc_init=gauss.json```                       |
| Figure 6, Leaky ReLU Activation  | ```depth_gt_rank=2_act=LeakyReLU_gnc_init=gauss.json```                  |
| Figure 8, Linear Activation      | ```depth_gt_rank=1_act=Linear_gnc_init=unif.json```                      |
| Figure 8, Tanh Activation        | ```depth_gt_rank=1_act=Tanh_gnc_init=unif.json```                        |
| Figure 8, Leaky ReLU Activation  | ```depth_gt_rank=1_act=LeakyReLU_gnc_init=unif.json```                   |
| Figure 10, Linear Activation     | ```depth_gt_rank=1_act=Linear_gnc_init=gauss_completion.json```          |
| Figure 10, Tanh Activation       | ```depth_gt_rank=1_act=Tanh_gnc_init=gauss_completion.json```            |
| Figure 10, Leaky ReLU Activation | ```depth_gt_rank=1_act=LeakyReLU_gnc_init=gauss_completion.json```       |
| Figure 11, Linear Activation     | ```depth_gt_rank=1_act=Linear_gnc_init=gauss_normalize=False.json```     |
| Figure 11, Tanh Activation       | ```depth_gt_rank=1_act=Tanh_gnc_init=gauss_normalize=False.json```       |
| Figure 11, Leaky ReLU Activation | ```depth_gt_rank=1_act=LeakyReLU_gnc_init=gauss_normalize=False.json```  |



## Citation

For citing the paper you can use:

```
@article{Alexander2025do,
  title={Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study},
  author={Alexander, Yotam and Slutzky, Yonatan and Ran-Milo, Yuval and Cohen, Nadav},
  journal={arXiv preprint arXiv:2506.03931},
  year={2025}
}
```
