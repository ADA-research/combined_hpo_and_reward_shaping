# Combined Optimisation of Hyperparameters and Reward Shape in Reinforcement Learning

<p align="center">
    <a href="./.logo.png#gh-light-mode-only">
        <img src="./.logo.png#gh-light-mode-only" alt="HPO+RSHAPE Logo" width="60%"/>
    </a>
    <a href="./.logo.png#gh-dark-mode-only">
        <img src="./.logo.png#gh-dark-mode-only" alt="HPO+RSHAPE Logo" width="60%"/>
    </a>
</p>


This repository contains supplementary material for our [RLC 2024 paper](https://arxiv.org/abs/2406.18293) on the combined optimisation of hyperparameters and reward shape in reinforcement learning. Below is an overview of the repository's contents:

- `source`: Contains the source code used for running our optimisation experiments, along with their corresponding configurations.
- `experimental_results/landscape_analysis`: Includes the landscape plots from the paper for the landscape analysis and the raw results of the performed reinforcement learning trainings. 
- `experimental_results/optimisation_experiments`: Includes the performance plots of our different experiments and datasets for incumbent performance and evaluation results of each experiment. 
- `behaviour_comparison`: Features videos comparing the behaviour of our optimised agents to baseline agents.

---

## Source Files and Parameter Configurations for the Optimisation Experiments

In the directory `source` we provide the source code and parameter configurations used in our optimisation experiments across the suites (i.e. Brax, Gym, and Robosuite) and environments (i.e. LunarLander, Ant, Humanoid, and Wipe). For the optimisation experiments with DEHB, we utilised the DEHB implementation as a Hydra sweeper, which can be found in this repository: [https://github.com/facebookresearch/how-to-autorl](https://github.com/facebookresearch/how-to-autorl).

- `config_suite_env`: Contains configuration files for the optimisation experiments for each suite and environment. Each directory includes the following two subdirectories:
  - `algorithm`: Provides the baseline's parameter configurations for SAC and PPO, including hyperparameters and reward shapes.
  - `search_space`: Defines the search spaces for the parameters in our different optimisation experiments.
- `dehb_suite_env`: Contains the reinforcement learning training implementation for each suite and environment. These implementations are utilised by the black-box optimisation algorithm DEHB for assessing the performance of a parameter configuration.
- `envs`: Includes helper functions and scripts for the LunarLander and Wipe environments to compute the task objectives and allow agent behaviour recording.

The conda environment used for our trainings is given in `environment.yml`. For installing the Hydra sweepers consider [https://github.com/facebookresearch/how-to-autorl](https://github.com/facebookresearch/how-to-autorl) together with the Hydra docs [https://hydra.cc](https://hydra.cc).

You can use the Hydra command line to run an experiment for PPO and SAC. For example, to optimise Gymnasium LunarLander with hyperparameter and reward weight optimisation, run:
```bash
python3 dehb_gym_lunar/ppo_hydra.py search_space=ppo_hpo_rshape
python3 dehb_gym_lunar/sac_hydra.py search_space=sac_hpo_rshape
```
The alternative search spaces of our experiments can be selected from `config_gym_lunar/search_spaces`. To perform a multi-objective optimisation with weighting the return and the stability in the final objective, simply run:
```bash
python3 dehb_gym_lunar/ppo_hydra.py search_space=ppo_hpo_rshape python3 sac_hydra.py search_space=sac_hpo_rshape algorithm.return_weight=0.5 algorithm.stability_weight=0.5
python3 dehb_gym_lunar/sac_hydra.py search_space=sac_hpo_rshape python3 sac_hydra.py search_space=sac_hpo_rshape algorithm.return_weight=0.5 algorithm.stability_weight=0.5
```


## Behaviour Comparison between Baseline Agents and Combined Optimisation of Hyperparameters and Reward Shape

The `behaviour_comparison` directory contains videos comparing the behaviour of our baseline agents to the behaviour of agents with jointly tuned hyperparameters and reward parameters. The comparison is based on the optimisation run that achieved median performance out of the five runs in the combined optimisation experiment. We then evaluated the best, median, and worst policies from the ten evaluation seeds of the baseline and the chosen optimisation run. In each video, we recorded the interaction of an agent with the environment for five episodes. The agent on the left is always the baseline agent, while the agent on the right is our tuned agent.


## Landscape Analysis Experimental Results 

We give the landscape experimental results in the directory `experimental_results/landscape_analysis`. The landscapes in the PDF files are similar to the ones found in the paper's appendix.

- `landscape_algmin.pdf`: Landscapes of the various hyperparameter and reward weight pairs, with the blue line indicating the optimal hyperparameter value for each reward weight and the black dots showing the configurations found in our optimisation experiments.
- `landscape_rewmin.pdf`: Landscapes of the various hyperparameter and reward weight pairs, with the blue line indicating the optimal reward weight for each hyperparameter value and the black dots showing the configurations found in our optimisation experiments.
- `hyperparameter_reward-weight.csv`: CSV files containing the raw data of the landscapes, thus containing the PPO LunarLander training t results for each configuration pair and training seed.


## Optimisation Experiment Results

We give the different results of our optimisation experiments for PPO and SAC in the directory `experimental_results/optimisation_experiments`. All plots are similar to the ones presented in the paper's appendix.

- `boxplots_separated.pdf`: Present the boxplots results over the 5 median performances of every optimisation experiment's runs.
- `boxplots_aggregated.pdf`: Present the boxplots results over the aggregated 50 evaluation trainings of every optimisation experiment.
- `incumbent_performance.pdf`: Present the median incumbent performance for each experiment's optimisation step.
- `optimisation_evaluation_results.csv`: Dataset of all the evaluation results for each experiment, optimisation seed, and evaluation seed.
- `optimisation_incumbent_data.csv`: Dataset of the incumbent performance of all experiments for each optimisation step and the final best-performing incumbent parameter configuration.


## Cite Us

```bibtex
@misc{DieEtAl24,
  author    = {Julian Dierkes and Emma Cramer and Holger H. Hoos and Sebastian Trimpe},
  title     = {Combining Automated Optimisation of Hyperparameters and Reward Shape},
  journal   = {In Proceedings of the 1st Reinforcement Learning Journal},
  volume    = 1, 
  year      = {2024},
  pages     = {1--9},
  url = {https://github.com/ADA-research/combined_hpo_and_reward_shaping},
}
```

