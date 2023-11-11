import collections
import argparse
import envs.crafter as crafter
import wandb
import functools
import os
import pathlib
import sys
from envs.crafter import targets, aware, reward_type_reverse
from tools import SliceDataset, get_episode_name, thresholds, lava_collect_limit

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
        self._step = int(count_steps(config.traindir) / 2)
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
                types = ["explore", "navigate"]
                for prefix in ["train", "eval"]:
                    for t in types:
                        total_successes = 0
                        total_failures = 0
                        for target_name in targets:
                            success_name = "{}_{}_success/{}".format(prefix, t, target_name)
                            successes = self._metrics.get(success_name, 0)
                            total_successes += successes
                            failure_name = "{}_{}_failure/{}".format(prefix, t, target_name)
                            failures = self._metrics.get(failure_name, 0)
                            total_failures += failures
                            if successes != 0 or failures != 0:
                                name = "{}_{}_success_rate/{}".format(prefix, t, target_name)
                                metrics_dict[name] = float(successes) / (failures + successes)
                                if t == "navigate":
                                    for subtype in ["face", "touch"]:
                                        subtype_success_name = "{}_{}_success/{}".format(prefix, subtype, target_name)
                                        subtype_name = "{}_{}_success_rate/{}".format(prefix, subtype, target_name)
                                        metrics_dict[subtype_name] = float(self._metrics.get(subtype_success_name, 0)) / (failures + successes)
                                        self._metrics.pop(subtype_success_name, None)
                            self._metrics.pop(success_name, None)
                            self._metrics.pop(failure_name, None)
                        if total_successes != 0 or total_failures != 0:
                            metrics_dict["{}_{}_success_rate/total".format(prefix, t)] = \
                                float(total_successes) / (total_failures + total_successes)
                    if self._metrics.get("{}_death_count".format(prefix), 0) != 0:
                        metrics_dict["{}_lava_death_rate".format(prefix)] = \
                            float(self._metrics.get("{}_lava_count".format(prefix), 0)) / self._metrics.get("{}_death_count".format(prefix))
                for name, values in self._metrics.items():
                    metrics_dict[name] = float(np.nanmean(values))
                for i in range(len(targets)):
                    metrics_dict["navigate_dataset_size/success_" + targets[i]] = self.navigate_dataset.success_aggregate_sizes[i]
                    metrics_dict["navigate_dataset_size/failure_" + targets[i]] = self.navigate_dataset.failure_aggregate_sizes[i]
                for i in range(len(targets)):
                    metrics_dict["explore_dataset_size/success_" + targets[i]] = self.explore_dataset.success_aggregate_sizes[i]
                    metrics_dict["explore_dataset_size/failure_" + targets[i]] = self.explore_dataset.failure_aggregate_sizes[i]
                metrics_dict["navigate_dataset_size/lava"] = sum([end - start for (start, end) in self.navigate_dataset.lava_deaths.values()])
                metrics_dict["explore_dataset_size/lava"] = sum([end - start for (start, end) in self.explore_dataset.lava_deaths.values()])
                openl = self._wm.video_pred(next(self._dataset))
                # 64 (64 * 3) (64 * 6) 3
                video = to_np(openl).transpose(0, 3, 1, 2)
                wandb.log({
                    "train_comp": wandb.Video(video, caption="train_comp", fps=10)
                })
                wandb.log(metrics_dict, step=step)
                self._metrics.clear()
        for i in range(len(obs["reward"])):
            types = ["explore", "navigate", "face", "touch"]
            for t in types:
                if obs["target_{}_steps".format(t)][i] >= 0:
                    target_name = targets[obs["prev_target"][i]]
                    step_name = "{}_{}_step/{}".format(mode, t, target_name)
                    success_name = "{}_{}_success/{}".format(mode, t, target_name)
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
        prev_target_array = torch.zeros((len(obs["image"])), dtype=torch.int32).to(self._config.device)
        for i, target in enumerate(obs["prev_target"]):
            prev_target_array[i] = target.to(self._config.device)
        target_array = torch.zeros((len(obs["image"])), dtype=torch.int32).to(self._config.device)
        for i, target in enumerate(obs["target"]):
            target_array[i] = target.to(self._config.device)
        stoch, deter = self._wm.dynamics.get_sep(latent)
        crafter_env = self.train_crafter if training else self.eval_crafter
        if obs["target_spot"] == 0:
            reward_prediction = self._wm.heads["navigate/reward"](stoch.unsqueeze(0), deter.unsqueeze(0), prev_target_array)
            means, policy_params = self._task_behavior.a2c_navigate(stoch, deter, target_array)
            crafter_env.reward_type = "navigate"
        else:
            reward_prediction = self._wm.heads["explore/reward"](stoch.unsqueeze(0), deter.unsqueeze(0), prev_target_array)
            means, policy_params = self._task_behavior.a2c_explore(stoch, deter, target_array)
            crafter_env.reward_type = "explore"
        where_prediction, front_prediction = self._wm.embed_where(embed)
        crafter_env.predicted_where = where_prediction.mode().reshape((len(aware), 4)).to(torch.long).cpu().detach().numpy().astype(np.uint8)
        crafter_env.predicted_front = front_prediction.mode().argmax().cpu().detach().numpy().astype(np.int8)
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
        return policy_output, state, tools.TwoHotDistSymlog(logits=means, device=self._config.device).mode().reshape(-1).item(), reward_prediction.mode().reshape(-1).item()

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
        task, outdir={"train": config.traindir, "eval": config.evaldir}[mode]
    )
    env = wrappers.OneHotAction(crafter_env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
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
    env = wrappers.CollectDataset(env, crafter_env, navigate_dataset, explore_dataset, callbacks=callbacks, directory=config.traindir)
    env = wrappers.RewardObs(env)
    return env, crafter_env


def load_slices(train_eps, navigate_dataset, explore_dataset):
    # Last augmented frame
    for ep_name, episode in train_eps.items():
        begin = 0
        target_spot = episode["target_spot"]
        target = episode["target"]
        print(target)
        for i in range(1, len(target)):
            if target_spot[i] != target_spot[i - 1] or target[i] != target[i - 1]:
                dataset = [navigate_dataset, explore_dataset][target_spot[i - 1]]
                step_name = ["target_navigate_steps", "target_explore_steps"][target_spot[i - 1]]
                is_success = episode[step_name][i] >= 0
                tuples = dataset.success_tuples if is_success else dataset.failure_tuples
                episode_sizes = dataset.success_episode_sizes if is_success else dataset.failure_episode_sizes
                aggregate_sizes = dataset.success_aggregate_sizes if is_success else dataset.failure_aggregate_sizes
                end = i + 1
                threshold = thresholds[["navigate", "explore"][target_spot[i - 1]]]
                target = episode["prev_target"][i]
                if end - begin >= threshold[target]:
                    if ep_name not in tuples[target]:
                        tuples[target][ep_name] = []
                        episode_sizes[target][ep_name] = 0
                    tuples[target][ep_name].append([begin, end])
                    episode_sizes[target][ep_name] += end - begin
                    aggregate_sizes[target] += end - begin
                begin = i
        dataset = [navigate_dataset, explore_dataset][episode["reward_mode"][-1]]
        cache = dataset.failure_tuples
        if ep_name not in cache[target[-1]]:
            cache[target[-1]][ep_name] = []
            dataset.failure_episode_sizes[target[-1]][ep_name] = 0
        if reward_type_reverse[episode["reward_type"][-1]] == "lava":
            start = len(target) - (lava_collect_limit - 1)
            for i in range(len(target) - 1, max(-1, len(target) - lava_collect_limit), -1):
                if np.sum(episode["where"][-1][aware.index("lava")]) == 0:
                    start = i + 1
                    break
            dataset.lava_deaths[ep_name] = (start, len(target))

        cache[target[-1]][ep_name].append([begin, len(target)])
        dataset.failure_episode_sizes[target[-1]][ep_name] += len(target) - begin
        dataset.failure_aggregate_sizes[target[-1]] += len(target) - begin
        navigate_dataset.save()
        explore_dataset.save()
        navigate_dataset.sanity_check()
        explore_dataset.sanity_check()

class ProcessEpisodeWrap:
    eval_scores = []
    eval_lengths = []
    last_step_at_eval = -1
    eval_done = False
    last_episode = 0

    @classmethod
    def process_episode(cls, config, logger, mode, train_eps, eval_eps, navigate_dataset, explore_dataset, episode):
        # this saved episodes is given as train_eps from next call
        filename = tools.save_episodes(config.traindir, [episode])[0]
        length = len(episode["reward"]) - 1
        score = float(episode["reward"].astype(np.float64).sum())
        video = episode["augmented"]
        train_eps[str(filename)] = episode
        if mode == "eval":
            eval_eps[str(filename)] = episode
        video = video[None].squeeze(0).transpose(0, 3, 1, 2)
        total = 0
        for key, ep in reversed(sorted(train_eps.items(), key=lambda x: x[0])):
            if not config.dataset_size or total <= config.dataset_size - length:
                total += len(ep["reward"]) - 1
            else:
                del train_eps[key]
                for dataset in [navigate_dataset, explore_dataset]:
                    for target_tuples in dataset.success_tuples:
                        target_tuples.pop(key, None)
                    for target_tuples in dataset.failure_tuples:
                        target_tuples.pop(key, None)
                    for i, target_sizes in enumerate(dataset.success_episode_sizes):
                        dataset.success_aggregate_sizes[i] -= target_sizes.get(key, 0)
                        target_sizes.pop(key, None)
                    for i, target_sizes in enumerate(dataset.failure_episode_sizes):
                        dataset.failure_aggregate_sizes[i] -= target_sizes.get(key, 0)
                        target_sizes.pop(key, None)
                    dataset.lava_deaths.pop(key, None)
                    dataset.save()
        if mode == "train":
            if wandb.run is not None:
                wandb.log({"dataset_size": total}, step=logger.step)
                if logger.step - cls.last_episode >= config.log_every:
                    cls.last_episode = logger.step
                    wandb.log({
                        f"{mode}_video": wandb.Video(video, caption=f"{mode}_video", fps=10)
                    }, step=logger.step)
            print(f"[{logger.step}] {mode.title()} episode has {length} steps and return {score:.1f}.")
            if wandb.run is not None and score > -10:
                wandb.log({f"{mode}_return": score, f"{mode}_length": length,
                           f"{mode}_episodes": len(train_eps)},
                          step=logger.step)
        elif mode == "eval":
            # keep only last 2 items for saving memory
            while len(eval_eps) > 2:
                # FIFO
                eval_eps.popitem()

            # start counting scores for evaluation
            if cls.last_step_at_eval != logger.step:
                cls.eval_scores = []
                cls.eval_lengths = []
                cls.eval_done = False
                cls.last_step_at_eval = logger.step

            cls.eval_scores.append(score)
            cls.eval_lengths.append(length)
            print(f"[{logger.step}] {mode.title()} episode has {length} steps and return {score:.1f}.")
            if wandb.run is not None and score > -10:
                wandb.log({f"{mode}_return": score, f"{mode}_length": length},
                          step=logger.step + len(cls.eval_lengths))
            # ignore if number of eval episodes exceeds eval_episode_num
            if len(cls.eval_scores) < config.eval_episode_num or cls.eval_done:
                return
            wandb.log({
                f"{mode}_video": wandb.Video(video, caption=f"{mode}_video", fps=10)
            }, step=logger.step)
            cls.eval_done = True


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
    last_counter = 0
    for k in train_eps.keys():
        st_idx = k.index("eps/") + len("eps/")
        ed_idx = k.index(".npz")
        last_counter = max(last_counter, int(k[st_idx:ed_idx]))
    get_episode_name.counter = last_counter + 1
    navigate_dataset = SliceDataset(train_eps, int(config.batch_size * 3/4), config.batch_length,
                                    str(Path.joinpath(directory, "navigate.json").absolute()), config.device,
                                    mode="train", name="navigate", ratio=config.success_failure_ratio)
    explore_dataset = SliceDataset(train_eps, int(config.batch_size * 1/4), config.batch_length,
                                   str(Path.joinpath(directory, "explore.json").absolute()), config.device,
                                   mode="train", name="explore", ratio=config.success_failure_ratio)
    load_slices(train_eps, navigate_dataset, explore_dataset)

    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    eval_dataset = make_dataset(eval_eps, config)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, navigate_dataset, explore_dataset)
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
            if o["target_spot"] == 0:
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
    watched = [(agent._wm.heads["explore/reward"], "explore_reward.", 6000),
               (agent._wm.heads["navigate/reward"], "navigate_reward.", 6000),
               (agent._wm.heads["image"], "image.", 3000),
               (agent._task_behavior.a2c_navigate, "a2c_navigate.", 15000),
               (agent._task_behavior.a2c_explore, "a2c_explore.", 15000)]
    wand_id = config.wandb_id or None
    resume = config.resume or None
    with wandb.init(project='mastering crafter with world models', config=defaults, id=wand_id, resume=resume):
    # with wandb.init(project='mastering crafter with world models', config=defaults):
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
            video = to_np(video_pred).transpose(0, 3, 1, 2)
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
