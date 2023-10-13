import collections
import argparse
import envs.crafter as crafter
import wandb
import functools
import os
import pathlib
import sys
from envs.crafter import targets
from tools import SliceDataset

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import envs.wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd
from pathlib import Path


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, config, logger, dataset, navigate_dataset, explore_dataset, train_crafter, eval_crafter):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_emit = tools.Every(config.emit_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = count_steps(config.traindir)
        self._update_count = 0
        self.train_crafter = train_crafter
        self.eval_crafter = eval_crafter
        # Schedules.
        config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(
            x, self._step
        )
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._step)
        )
        config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(
            x, self._step
        )
        self._dataset = dataset
        self.navigate_dataset = navigate_dataset
        self.explore_dataset = explore_dataset
        self._wm = models.WorldModel(self._step, config, self.navigate_dataset, self.explore_dataset)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if config.compile:
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

    def __call__(self, obs, reset, state=None, reward=None, training=True):
        step = self._step
        mode = "train" if training else "eval"
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train()
                self._update_count += 1
                self._metrics["update_count"] = self._update_count

            if self._should_log(step):
                metrics_dict = {}
                for prefix in ["train", "eval"]:
                    for t in types:
                        total_successes = 0
                        total_failures = 0
                        for target_name in targets:
                            success_name = prefix + "_" + target_name + "_{}_success".format(t)
                            successes = self._metrics.get(success_name, 0)
                            total_successes += successes
                            failure_name = prefix + "_" + target_name + "_{}_failure".format(t)
                            failures = self._metrics.get(failure_name, 0)
                            total_failures += failures
                            if successes != 0 or failures != 0:
                                name = prefix + "_" + target_name + "_{}_success_rate".format(t)
                                metrics_dict[name] = float(successes) / (failures + successes)
                            self._metrics.pop(success_name, None)
                            self._metrics.pop(failure_name, None)
                        if total_successes != 0 or total_failures != 0:
                            metrics_dict["total_" + prefix + "_{}_success_rate".format(t)] = \
                                float(total_successes) / (total_failures + total_successes)
                    if self._metrics.get("death_count", 0) != 0:
                        metrics_dict["lava_death_rate"] = float(self._metrics.get("lava_count", 0)) / self._metrics.get("death_count")
                for name, values in self._metrics.items():
                    metrics_dict[name] = float(np.nanmean(values))
                for i in range(len(targets)):
                    metrics_dict[targets[i] + "_navigate_dataset_size"] = self.navigate_dataset.aggregate_sizes[i]
                for i in range(len(targets)):
                    metrics_dict[targets[i] + "_explore_dataset_size"] = self.explore_dataset.aggregate_sizes[i]
                openl = self._wm.video_pred(next(self._dataset))
                # 6 64 192 64 3
                video = to_np(openl[0]).transpose(0, 3, 1, 2)
                wandb.log({
                    "train_comp": wandb.Video(video, caption="train_comp", fps=10)
                })
                wandb.log(metrics_dict, step=step)
                self._metrics.clear()
        for i in range(len(obs["reward"])):
            types = ["spot", "reached"]
            for t in types:
                if obs["target_{}_steps".format(t)][i] >= 0:
                    target_name = targets[obs["prev_target"][i]]
                    step_name = mode + "_" + target_name + "_{}_step".format(t)
                    success_name = mode + "_" + target_name + "_{}_success".format(t)
                    if step_name not in self._metrics.keys():
                        self._metrics[step_name] = [obs["target_{}_steps".format(t)][i]]
                        self._metrics[success_name] = 1
                    else:
                        self._metrics[step_name].append(obs["target_{}_steps".format(t)][i])
                        self._metrics[success_name] += 1

        policy_output, state, value, reward = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state, value, reward

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        target_array = torch.zeros((len(obs["image"])), dtype=torch.int32).to(self._config.device)
        for i, target in enumerate(obs["target"]):
            target_array[i] = target.to(self._config.device)
        stoch, deter = self._wm.dynamics.get_sep(latent)
        crafter_env = self.train_crafter if training else self.eval_crafter
        if obs["target_spot"]:
            reward_prediction = self._wm.heads["navigate_reward"](stoch.unsqueeze(0), deter.unsqueeze(0), target_array)
            means, policy_params = self._task_behavior.a2c_navigate(stoch, deter, target_array)
            crafter_env.reward_type = "navigate"
        else:
            reward_prediction = self._wm.heads["explore_reward"](stoch.unsqueeze(0), deter.unsqueeze(0), target_array)
            means, policy_params = self._task_behavior.a2c_explore(stoch, deter, target_array)
            crafter_env.reward_type = "explore"
        actor = tools.OneHotDist(policy_params, unimix_ratio=self._config.action_unimix_ratio)
        if not training:
            action = actor.mode()
        else:
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state, tools.TwoHotDistSymlog(logits=means).mode().reshape(-1).item(), reward_prediction.mode().reshape(-1).item()

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        probs = amount / self._config.num_actions + (1 - amount) * action
        return tools.OneHotDist(probs=probs).sample()

    def _train(self):
        metrics = {}
        navigate_post, explore_post, navigate_data, explore_data, mets = self._wm._train()
        metrics.update(mets)
        # start['deter'] (16, 64, 512)
        metrics.update(self._task_behavior._train(navigate_post, explore_post, navigate_data, explore_data)[-1])
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    total = 0
    for filename in reversed(sorted(folder.glob("*.npz"))):
        try:
            with filename.open("rb") as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f"Could not load episode: {e}")
            continue
        total += len(episode["reward"]) - 1
    return total


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, logger, mode, train_eps, eval_eps, navigate_dataset, explore_dataset):
    # crafter_reward
    # [crafter, reward]
    suite, task = config.task.split("_", 1)
    crafter_env = crafter.Crafter(
        task, outdir="./stats"
    )
    env = wrappers.OneHotAction(crafter_env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    if (mode == "train") or (mode == "eval"):
        callbacks = [
            functools.partial(
                ProcessEpisodeWrap.process_episode,
                config,
                logger,
                mode,
                train_eps,
                eval_eps,
                navigate_dataset,
                explore_dataset
            )
        ]
        dir = dict(train=config.traindir, eval=config.evaldir)[mode]
        eps = dict(train=train_eps, eval=eval_eps)[mode]
        env = wrappers.CollectDataset(env, eps, navigate_dataset, explore_dataset, callbacks=callbacks, directory=dir)
    env = wrappers.RewardObs(env)
    return env, crafter_env


class ProcessEpisodeWrap:
    eval_scores = []
    eval_lengths = []
    last_step_at_eval = -1
    eval_done = False
    last_episode = 0

    @classmethod
    def process_episode(cls, config, logger, mode, train_eps, eval_eps, navigate_dataset, explore_dataset, episode):
        directory = dict(train=config.traindir, eval=config.evaldir)[mode]
        cache = dict(train=train_eps, eval=eval_eps)[mode]
        # this saved episodes is given as train_eps or eval_eps from next call
        filename = tools.save_episodes(directory, [episode])[0]
        length = len(episode["reward"]) - 1
        score = float(episode["reward"].astype(np.float64).sum())
        video = episode["augmented"]
        cache[str(filename)] = episode
        video = video[None].squeeze(0).transpose(0, 3, 1, 2)
        if mode == "train":
            total = 0
            for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
                if not config.dataset_size or total <= config.dataset_size - length:
                    total += len(ep["reward"]) - 1
                else:
                    del cache[key]
                    for dataset in [navigate_dataset, explore_dataset]:
                        for target_tuples in dataset.tuples:
                            target_tuples.pop(key, None)
                        for i, target_sizes in enumerate(dataset.episode_sizes):
                            dataset.aggregate_sizes[i] -= target_sizes.get(key, 0)
                            target_sizes.pop(key, None)
                        dataset.save()
            if wandb.run is not None:
                wandb.log({"dataset_size": total}, step=logger.step)
                if logger.step - cls.last_episode >= config.log_every:
                    cls.last_episode = logger.step
                    wandb.log({
                        f"{mode}_video": wandb.Video(video, caption=f"{mode}_video", fps=10)
                    }, step=logger.step)
        elif mode == "eval":
            # keep only last item for saving memory
            while len(cache) > 1:
                # FIFO
                a = cache.popitem()

            # start counting scores for evaluation
            if cls.last_step_at_eval != logger.step:
                cls.eval_scores = []
                cls.eval_lengths = []
                cls.eval_done = False
                cls.last_step_at_eval = logger.step

            cls.eval_scores.append(score)
            cls.eval_lengths.append(length)
            # ignore if number of eval episodes exceeds eval_episode_num
            if len(cls.eval_scores) < config.eval_episode_num or cls.eval_done:
                return
            score = sum(cls.eval_scores) / len(cls.eval_scores)
            length = sum(cls.eval_lengths) / len(cls.eval_lengths)
            episode_num = len(cls.eval_scores)
            wandb.log({
                f"{mode}_video": wandb.Video(video, caption=f"{mode}_video", fps=10)
            }, step=logger.step)
            cls.eval_done = True

        print(f"[{logger.step}] {mode.title()} episode has {length} steps and return {score:.1f}.")
        if wandb.run is not None and score > -5:
            wandb.log({f"{mode}_return": score, f"{mode}_length": length, f"{mode}_episodes": len(cache) if mode == "train" else episode_num}, step=logger.step)


def main(config, defaults):
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)
    config.norm = getattr(torch.nn, config.norm)

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    train_dataset = make_dataset(train_eps, config)
    navigate_dataset = SliceDataset(train_eps, config.batch_size, config.batch_length,
                                    str(Path.joinpath(directory, "navigate.json").absolute()), name="train")
    explore_dataset = SliceDataset(train_eps, config.batch_size, config.batch_length,
                                   str(Path.joinpath(directory, "explore.json").absolute()), name="train")

    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    eval_dataset = make_dataset(eval_eps, config)
    navigate_dataset_eval = SliceDataset(eval_eps, config.batch_size, config.batch_length,
                                    str(Path.joinpath(directory, "navigate.json").absolute()), name="eval")
    explore_dataset_eval = SliceDataset(eval_eps, config.batch_size, config.batch_length,
                                   str(Path.joinpath(directory, "explore.json").absolute()), name="eval")
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, navigate_dataset if mode == "train" else navigate_dataset_eval,
                                 explore_dataset if mode == "train" else explore_dataset_eval)
    train_env, train_crafter = make("train")
    eval_env, eval_crafter = make("eval")
    acts = train_env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s, r, training=True):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            crafter_env = train_crafter if training else eval_crafter
            if o["target_spot"]:
                crafter_env.reward_type = "navigate"
            else:
                crafter_env.reward_type = "explore"
            return {"action": action, "logprob": logprob}, None, 0, 0

        tools.simulate(random_agent, train_env, train_crafter, prefill)
        logger.step = config.action_repeat * count_steps(config.traindir)

    print("Simulate agent.")
    agent = Dreamer(config, logger, train_dataset, navigate_dataset, explore_dataset, train_crafter, eval_crafter).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest_model.pt").exists():
        agent.load_state_dict(torch.load(logdir / "latest_model.pt"))
        agent._should_pretrain._once = False

    state = None
    watched = [(agent._wm.heads["explore_reward"], "explore_reward.", 6000),
               (agent._wm.heads["navigate_reward"], "navigate_reward.", 6000),
               (agent._wm.heads["image"], "image.", 2000),
               (agent._task_behavior.a2c_navigate, "a2c_navigate.", 15000),
               (agent._task_behavior.a2c_explore, "a2c_explore.", 15000)]
    # with wandb.init(project='mastering crafter with world models', config=defaults, id="nf5cv15k", resume=True):
    with wandb.init(project='mastering crafter with world models', config=defaults):
        for model, name, param_freq in watched:
            model.requires_grad_(requires_grad=True)
            wandb.run._torch.add_log_parameters_hook(
                model,
                prefix=name,
                log_freq=param_freq,
            )
            wandb.run._torch.add_log_gradients_hook(
                model,
                prefix=name,
                log_freq=1000,
            )
            model.requires_grad_(requires_grad=False)

        while agent._step < config.steps:
            print("Start training.")
            state = tools.simulate(agent, train_env, train_crafter, config.eval_every, state=state, metrics=agent._metrics)
            torch.save(agent.state_dict(), logdir / "latest_model.pt")
            print("Start evaluation.")
            tools.simulate(agent, eval_env, eval_crafter, episodes=config.eval_episode_num, training=False, metrics=agent._metrics)
            video_pred = agent._wm.video_pred(next(eval_dataset))
            video = to_np(video_pred[0]).transpose(0, 3, 1, 2)
            wandb.log({
                "eval_comp": wandb.Video(video, caption="eval_comp", fps=10)
            })
    for env in [train_env, eval_env]:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    torch.set_printoptions(linewidth=150, precision=2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    defaults = {}
    wandb.login()
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining), defaults)
