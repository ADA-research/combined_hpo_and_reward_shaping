import os
import logging

import numpy as np
import hashlib

from rollout_callback import LogAfterRollout

from sbx import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from gymnasium.wrappers import TimeLimit
from dehb_gym_lunar.lunar_lander import LunarLander

import hydra
from omegaconf import OmegaConf


time_limit = 1000


def external_policy_evaluation(
    model, env, n_eval_episodes, deterministic=True, render=False,
):
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if dones[i]:
                    if "episode" in info.keys():
                        if info["success"]:
                            episode_rewards.append(info["episode"]["l"])
                            episode_lengths.append(info["episode"]["l"])
                        else:
                            episode_rewards.append(time_limit)
                            episode_lengths.append(info["episode"]["l"])
                        episode_counts[i] += 1

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


class EvalTrainingMetricsCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq,
        n_eval_episodes,
        return_weight=1.0,
        stability_weight=0.0,
        log_wandb=False,
        wandb_run=None
    ):
        super().__init__(verbose=0)

        self.best_performance = np.inf
        self.best_eval_return = None
        self.best_eval_return_std = None
        self.best_model_params = None

        self.return_weight = return_weight
        self.stability_weight = stability_weight

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.log_wandb = log_wandb
        self.wandb_run = wandb_run

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_return, std_return = external_policy_evaluation(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )
            performance = self.return_weight * mean_return + self.stability_weight * std_return
            if performance < self.best_performance:
                self.best_performance = performance
                self.best_eval_return = mean_return
                self.best_eval_return_std = std_return
                self.best_model_params = (self.model.policy.actor_state, self.model.policy.vf_state)
            if self.log_wandb:
                self.wandb_run.log({"eval/mean_return": mean_return, "eval/std_return": std_return}, step=self.num_timesteps)

    def _on_training_end(self) -> None:
        mean_return, std_return = external_policy_evaluation(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
        )
        performance = self.return_weight * mean_return + self.stability_weight * std_return
        if performance < self.best_performance:
            self.best_performance = performance
            self.best_eval_return = mean_return
            self.best_eval_return_std = std_return
            self.best_model_params = (self.model.policy.actor_state, self.model.policy.vf_state)
        if self.log_wandb:
            self.wandb_run.log({"eval/mean_return": mean_return, "eval/std_return": std_return}, step=self.num_timesteps)

    def get_best(self):
        return self.best_performance, self.best_eval_return, self.best_eval_return_std, self.best_model_params


def get_random_rshape(cfg, param_name, rand_gen):
    config = cfg.search_space.hyperparameters[f"algorithm.reward_shape.{param_name}"]
    if config["log"]:
        raise NotImplementedError
    if config["type"] == "uniform_float":
        param = rand_gen.uniform(config["lower"], config["upper"])
    elif config["type"] == "uniform_int":
        param = rand_gen.randint(config["lower"], config["upper"])
    else:
        raise NotImplementedError
    return param


def hydra_launcher(cfg, log) -> float:
    if cfg.search_space.random_reward_weights:
        temp_eval = cfg.algorithm.eval
        temp_total_timesteps = cfg.algorithm.total_timesteps
        cfg.algorithm.eval = False  # dummy value
        cfg.algorithm.total_timesteps = 0  # dummy value
        # sample random_reward weights from a hash of the config
        rshape_rand_seed = np.frombuffer(hashlib.sha256(str(cfg.algorithm).encode('utf-8')).digest(), dtype=np.uint32)
        cfg.algorithm.eval = temp_eval
        cfg.algorithm.total_timesteps = temp_total_timesteps

        rshape_rand_gen = np.random.RandomState(rshape_rand_seed)
        cfg.algorithm.reward_shape["distance_rew_weight"] = get_random_rshape(cfg, "distance_rew_weight", rshape_rand_gen)
        cfg.algorithm.reward_shape["velocity_rew_weight"] = get_random_rshape(cfg, "velocity_rew_weight", rshape_rand_gen)
        cfg.algorithm.reward_shape["angle_rew_weight"] = get_random_rshape(cfg, "angle_rew_weight", rshape_rand_gen)
        cfg.algorithm.reward_shape["leg_rew_weight"] = get_random_rshape(cfg, "leg_rew_weight", rshape_rand_gen)
        log.info(f"rshape_rand_seed: {rshape_rand_seed}")

    cfg.algorithm.model_kwargs['gamma'] = 1 - cfg.algorithm.model_kwargs['gamma']
    cfg.algorithm.model_kwargs['seed'] = cfg.seed
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.algorithm.eval:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            job_type=cfg.wandb_type,
            config=dict(cfg),
            monitor_gym=True,
            sync_tensorboard=True,
        )
        run_id = run.id
        log_path = f"./work/oh751555/dehb_logs/{run_id}"
        run.log({"best_params_path": log_path})
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        save_path = f"./work/oh751555/dehb_logs/{run_id}"

    def make_train_env():
        def _init():
            return TimeLimit(LunarLander(
                    render_mode=None,
                    continuous=True,
                    distance_rew_weight=cfg.algorithm.reward_shape.distance_rew_weight,
                    angle_rew_weight=cfg.algorithm.reward_shape.angle_rew_weight,
                    velocity_rew_weight=cfg.algorithm.reward_shape.velocity_rew_weight,
                    leg_rew_weight=cfg.algorithm.reward_shape.leg_rew_weight,
                ),
                max_episode_steps=time_limit,
            )
        return _init

    def make_eval_env():
        def _init():
            return TimeLimit(LunarLander(
                    render_mode=None,
                    continuous=True,
                ),
                max_episode_steps=time_limit,
            )
        return _init

    def make_default_env():
        def _init():
            return TimeLimit(LunarLander(
                    render_mode=None,
                    continuous=True,
                ),
                max_episode_steps=time_limit,
            )
        return _init

    train_env = VecMonitor(DummyVecEnv([make_train_env() for _ in range(cfg.algorithm.num_envs)]))

    model = PPO(
        cfg.algorithm.policy_model,
        train_env,
        verbose=1 if cfg.algorithm.eval else 0,
        tensorboard_log=save_path if cfg.algorithm.eval else None,
        **cfg.algorithm.model_kwargs
    )

    eval_env = VecMonitor(DummyVecEnv([make_eval_env() for _ in range(cfg.algorithm.num_envs)]))
    eval_env.seed(cfg.seed+1)
    log_callback = LogAfterRollout(log_freq=1)
    eval_callback = EvalTrainingMetricsCallback(
        eval_env,
        n_eval_episodes=128,
        eval_freq=5000,
        return_weight=cfg.algorithm.return_weight,
        stability_weight=cfg.algorithm.stability_weight,
        log_wandb=cfg.algorithm.eval,
        wandb_run=run if cfg.algorithm.eval else None
    )
    model.learn(total_timesteps=cfg.algorithm.total_timesteps, callback=[log_callback, eval_callback], log_interval=None)
    best_performance, best_return, best_return_std, best_model_params = eval_callback.get_best()
    log.info(f"Mean evaluation return across 128 episodes was {best_return} with std {best_return_std}")
    
    if cfg.algorithm.eval:
        model.policy.actor_state = best_model_params[0]
        model.policy.vf_state = best_model_params[1]

        eval_env.seed(cfg.seed+2)
        eval_mean_return, eval_std_return = external_policy_evaluation(
            model, eval_env, n_eval_episodes=128
        )
        eval_performance = cfg.algorithm.return_weight * eval_mean_return + cfg.algorithm.stability_weight * eval_std_return
        run.log({
            f"{cfg.algorithm.return_weight}-{cfg.algorithm.stability_weight}_weighted_sum": eval_performance,
            "final_mean_return": eval_mean_return,
            "final_std_return": eval_std_return
        })
        default_env = VecMonitor(DummyVecEnv([make_default_env() for _ in range(cfg.algorithm.num_envs)]))
        default_env.seed(cfg.seed+2)
        def_rew_mean_return, def_rew_std_return = evaluate_policy(model, default_env, n_eval_episodes=128)
        run.log({"def_rew_mean_return": def_rew_mean_return, "def_rew_std_return": def_rew_std_return})

        def make_recording_env():
            def _init():
                return Monitor(TimeLimit(LunarLander(
                        render_mode="rgb_array",
                        continuous=True,
                    ),
                    max_episode_steps=time_limit,
                ))
            return _init
        record_env = DummyVecEnv([make_recording_env() for _ in range(1)])
        record_env.seed(cfg.seed+2)
        record_env = VecVideoRecorder(
            record_env,
            video_length=int(1e8),
            video_folder=os.path.join(save_path, "videos"),
            record_video_trigger=lambda x: True,
        )
        _, _ = evaluate_policy(model, record_env, n_eval_episodes=5, render=True)

    return best_performance


@hydra.main(version_base="1.3", config_path="../config_gym_lunar", config_name="local_ppo_runner")
def local_ppo_runner(cfg):
    try:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        return hydra_launcher(cfg, log)
    except Exception as e:
        log.error(e)
        return None

if __name__ == '__main__':
    local_ppo_runner()
