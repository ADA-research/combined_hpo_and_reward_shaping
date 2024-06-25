import os
import logging
import functools
from datetime import datetime

import numpy as np
import hashlib

import jax
import brax.envs
from brax.training.acting import Evaluator
from brax.io import model, html
from brax.training.agents.sac import train as sac

import hydra
from omegaconf import OmegaConf


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
    if cfg.algorithm.eval:
        cfg.algorithm.model_kwargs['num_evals'] = 10
    else:
        cfg.algorithm.model_kwargs['num_evals'] = 10 // (5242881 // cfg.algorithm['num_timesteps'])
    log.info(f"num_evals: {cfg.algorithm.model_kwargs['num_evals']}")

    if cfg.search_space.random_reward_weights:
        temp_eval = cfg.algorithm.eval
        temp_num_timesteps = cfg.algorithm.num_timesteps
        cfg.algorithm.eval = False  # dummy value
        cfg.algorithm.num_timesteps = 0  # dummy value
        # sample random_reward weights from a hash of the config
        rshape_rand_seed = np.frombuffer(hashlib.sha256(str(cfg.algorithm).encode('utf-8')).digest(), dtype=np.uint32)
        cfg.algorithm.eval = temp_eval
        cfg.algorithm.num_timesteps = temp_num_timesteps

        rshape_rand_gen = np.random.RandomState(rshape_rand_seed)
        cfg.algorithm.reward_shape["forward_reward_weight"] = get_random_rshape(cfg, "forward_reward_weight", rshape_rand_gen)
        cfg.algorithm.reward_shape["healthy_reward"] = get_random_rshape(cfg, "healthy_reward", rshape_rand_gen)
        cfg.algorithm.reward_shape["ctrl_cost_weight"] = get_random_rshape(cfg, "ctrl_cost_weight", rshape_rand_gen)
        log.info(f"rshape_rand_seed: {rshape_rand_seed}")

    # normalize by summed reward and multiply with 2.5 (default summed rewards) to obtain same scale as hand-tuned reward-shaping
    # referred to as "explicit scaling" in the paper
    if cfg.search_space.scale_reward:
        summed_reward_shapes = cfg.algorithm.reward_shape["forward_reward_weight"]
        summed_reward_shapes += cfg.algorithm.reward_shape["healthy_reward"]
        summed_reward_shapes += cfg.algorithm.reward_shape["ctrl_cost_weight"]
        cfg.algorithm.reward_shape["forward_reward_weight"] = (cfg.algorithm.reward_shape["forward_reward_weight"] * 2.5) / summed_reward_shapes
        cfg.algorithm.reward_shape["healthy_reward"] = (cfg.algorithm.reward_shape["healthy_reward"] * 2.5) / summed_reward_shapes
        cfg.algorithm.reward_shape["ctrl_cost_weight"] = (cfg.algorithm.reward_shape["ctrl_cost_weight"] * 2.5) / summed_reward_shapes

    cfg.algorithm.model_kwargs['discounting'] = 1 - cfg.algorithm.model_kwargs['discounting']
    cfg.algorithm.model_kwargs['seed'] = cfg.seed
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.algorithm.eval:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            job_type=cfg.wandb_type,
            config=dict(cfg)
        )
        run_id = run.id
        log_path = f"/work/oh751555/dehb_logs/{run_id}"
        run.log({"best_params_path": log_path})
        if not os.path.exists(log_path):
            os.makedirs(log_path)


    train_env = brax.envs.get_environment(
        env_name="ant",
        backend=cfg.algorithm.backend,
        forward_reward_weight=cfg.algorithm.reward_shape.forward_reward_weight,
        ctrl_cost_weight=cfg.algorithm.reward_shape.ctrl_cost_weight,
        healthy_reward=cfg.algorithm.reward_shape.healthy_reward,
    )

    eval_env = brax.envs.get_environment(
        env_name="ant",
        backend=cfg.algorithm.backend,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.0,
        healthy_reward=0.0,
    )

    train_fn = functools.partial(
        sac.train,
        num_timesteps=cfg.algorithm.num_timesteps,
        return_best_eval_rew_and_params=True,
        best_eval_reward_weight=cfg.algorithm.return_weight,
        best_eval_stability_weight=cfg.algorithm.stability_weight,
        reward_scaling=cfg.algorithm.reward_shape.reward_scaling if cfg.search_space.scale_reward else 1.0,
        **cfg.algorithm.model_kwargs,
    )

    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        if cfg.algorithm.eval:
            run.log({"rollout/episode_reward": metrics["eval/episode_reward"]}, step=1 if num_steps == 0 else num_steps)
        log.info({"rollout/episode_reward": metrics["eval/episode_reward"]})

    make_inference_fn, best_eval_params, _, best_eval_performance, best_eval_rew, best_eval_std = train_fn(
        environment=train_env,
        eval_env=eval_env,
        progress_fn=progress
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    if cfg.algorithm.eval:
        model.save_params(os.path.join(log_path, "best_params"), best_eval_params)
        run.log({
            f"{cfg.algorithm.return_weight}-{cfg.algorithm.stability_weight}_weighted_sum": best_eval_performance,
            "final_mean_return": best_eval_rew,
            "final_std_return": best_eval_std
        })

        inference_fn = make_inference_fn(best_eval_params, deterministic=True)
        jit_eval_env_reset = jax.jit(eval_env.reset)
        jit_eval_env_step = jax.jit(eval_env.step)
        jit_inference_fn = jax.jit(inference_fn)

        rollout = []
        rng = jax.random.PRNGKey(seed=cfg.seed+1)
        for episode in range(5):
            state = jit_eval_env_reset(rng=rng)
            for _ in range(1000):
                rollout.append(state.pipeline_state)
                act_rng, rng = jax.random.split(rng)
                act, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_eval_env_step(state, act)
                if state.done:
                    break

        html_vis = html.render(eval_env.sys.replace(dt=eval_env.dt), rollout)
        run.log({"visualization": wandb.Html(html_vis)})

        default_env = brax.envs.get_environment(env_name="ant",backend=cfg.algorithm.backend)
        default_env = brax.envs.wrappers.training.wrap(
            default_env,
            episode_length=1000,
            action_repeat=1,
        )
        evaluator = Evaluator(
            default_env,
            functools.partial(make_inference_fn, deterministic=True),
            num_eval_envs=128,
            episode_length=1000,
            action_repeat=1,
            key=jax.random.PRNGKey(seed=cfg.seed+1)
        )
        metrics = evaluator.run_evaluation(best_eval_params, {})
        run.log({
            "def_rew_mean_return": metrics["eval/episode_reward"],
            "def_rew_std_return": metrics["eval/episode_reward_std"]
        })
        run.finish()

    return -best_eval_performance


@hydra.main(version_base="1.3", config_path="../config_brax_ant", config_name="local_sac_runner")
def local_sac_runner(cfg):
    try:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        return hydra_launcher(cfg, log)
    except Exception as e:
        log.error(e)
        raise e


if __name__ == '__main__':
    local_sac_runner()
