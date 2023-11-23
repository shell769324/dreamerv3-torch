import copy

import torch
from torch import nn
import networks
import tools
from envs.crafter import targets, navigate_targets, combat_targets, aware
import numpy as np


def to_np(x):
    return x.detach().cpu().numpy()


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, step, config, navigate_dataset, explore_dataset, combat_dataset):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.navigate_dataset = navigate_dataset
        self.explore_dataset = explore_dataset
        self.combat_dataset = combat_dataset

        self.encoder = networks.ConvEncoder(
            config.grayscale,
            config.cnn_depth,
            config.act,
            config.norm,
            config.encoder_kernels
        )

        if config.size[0] == 64 and config.size[1] == 64:
            embed_size = ((64 // 2 ** (len(config.encoder_kernels))) ** 2 * config.cnn_depth * 2 **
                          (len(config.encoder_kernels) - 1))
        else:
            raise NotImplemented(f"{config.size} is not applicable now")

        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.unimix_ratio,
            config.num_actions,
            embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        channels = 1 if config.grayscale else 3
        shape = (channels,) + config.size
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        self.heads["image"] = networks.ConvDecoder(
            feat_size,  # pytorch version
            config.cnn_depth,
            config.act,
            config.norm,
            shape,
            config.decoder_kernels,
        )
        """self.heads["where"] = networks.DenseHead(
            feat_size,  # pytorch version
            (len(targets) * 5,),
            config.where_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )
        """
        self.heads["explore/reward"] = networks.EmbeddedDenseHead(
            self._config.dyn_stoch * self._config.dyn_discrete,
            config.dyn_deter,
            config.explore_reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
            embed_count=len(navigate_targets)
        )
        self.heads["navigate/reward"] = networks.EmbeddedDenseHead(
            self._config.dyn_stoch * self._config.dyn_discrete,
            config.dyn_deter,
            config.navigate_reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
            embed_count=len(navigate_targets)
        )
        self.heads["combat/reward"] = networks.EmbeddedDenseHead(
            self._config.dyn_stoch * self._config.dyn_discrete,
            config.dyn_deter,
            config.combat_reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
            embed_count=len(combat_targets)
        )
        self.heads["cont"] = networks.DenseHead(
            feat_size,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )
        self.embed_where = networks.WhereHead(
            12288,
            config.where_layers,
            config.units,
            config.act,
            config.norm,
            device=config.device,
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "world_model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
            sub={"cont": self.heads["cont"], "encoder": self.encoder,
                 "rssm": self.dynamics, "explore/reward": self.heads["explore/reward"], "navigate/reward": self.heads["navigate/reward"],
                 "combat/reward": self.heads["combat/reward"],
                 "where": self.embed_where, "image": self.heads["image"]
                 }#"where": self.heads["where"]}
        )
        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale, where=config.where_scale)

    def _train(self):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        metrics = {}
        threshold = torch.tensor(self._config.regularize_threshold).to(self._config.device)
        coeff = torch.tensor(self._config.regularization).to(self._config.device)
        sample_data = {}
        sample_markers = {}
        for name, dataset, specific_targets in [("navigate", self.navigate_dataset, navigate_targets), ("explore", self.explore_dataset, navigate_targets),
                        ("combat", self.combat_dataset, combat_targets)]:
            difficulty = np.array(dataset.failure_aggregate_sizes) / \
                         (1 + np.array(dataset.success_aggregate_sizes) + np.array(dataset.failure_aggregate_sizes))
            target_dist = difficulty / np.sum(difficulty)
            data, markers, lava_rate, success_dist, failure_dist = dataset.sample(target_dist)
            metrics["{}/lava_sample_rate".format(name)] = lava_rate
            for i, t in enumerate(specific_targets):
                metrics["{}/{}_success_sample_rate".format(name, t)] = success_dist[i]
                metrics["{}/{}_failure_sample_rate".format(name, t)] = failure_dist[i]
            data = self.preprocess(data)
            sample_data[name] = data
            sample_markers[name] = markers
        data = {k: torch.cat([v[k] for v in sample_data.values()], dim=0) for k in sample_data["navigate"].keys()}

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                losses = {}
                embeds = {k: self.encoder(v) for k, v in sample_data.items()}
                combined_embed = torch.cat(list(embeds.values()), 0)
                where_pred, front_pred = self.embed_where(combined_embed)
                where_like = where_pred.log_prob(data["where"].reshape(data["where"].shape[:-2] + (np.prod(data["where"].shape[-2:]),)))
                front_like = front_pred.log_prob(torch.nn.functional.one_hot(data["front"], num_classes=len(aware) + 1).to(self._config.device))
                losses["where"] = -torch.mean(where_like) * self._scales.get("where", 1.0)
                losses["front"] = -torch.mean(front_like) * self._scales.get("front", 1.0)
                posts = {}
                priors = {}
                for k, v in sample_data.items():
                    post, prior = self.dynamics.observe(
                        embeds[k], v["action"], v["is_first"], markers=sample_markers[k]
                    )
                    posts[k] = post
                    priors[k] = prior
                prior = {k: torch.cat([v[k] for v in priors.values()], dim=0) for k in priors["navigate"].keys()}
                post = {k: torch.cat([v[k] for v in posts.values()], dim=0) for k in posts["navigate"].keys()}
                kl_free = tools.schedule(self._config.kl_free, self._step)
                dyn_scale = tools.schedule(self._config.dyn_scale, self._step)
                rep_scale = tools.schedule(self._config.rep_scale, self._step)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                for name, head in self.heads.items():
                    if "reward" not in name:
                        feat = self.dynamics.get_feat(post)
                        pred = head(feat)
                        like = pred.log_prob(data[name])

                        losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)

                    if "reward" in name:
                        actor_mode = name.split('/')[0]
                        right_data = sample_data[actor_mode]
                        # Detach explore post since it doesn't add anything to world model
                        right_post = posts[actor_mode]
                        target_name = "prev_{}_target".format(actor_mode if actor_mode == "combat" else "navigate")
                        (stoch, deter) = self.dynamics.get_sep(right_post)
                        pred = head(stoch, deter, right_data[target_name])
                        like = pred.log_prob(right_data["reward"])
                        # Ignore the first slice of each segment
                        like = like[sample_markers[actor_mode] == 0]
                        losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                        reward_logits = pred.logits.abs()
                        reward_suppressor = (reward_logits[reward_logits > threshold] - threshold).mean() * coeff
                        if not reward_suppressor.isnan().any():
                            losses[name] += reward_suppressor
                        metrics.update(tools.tensorstats(pred.logits, "{}_logits".format(name)))
                        metrics["{}_suppressor".format(name)] = to_np(reward_suppressor)

                model_loss = sum(losses.values()) + kl_loss
                metrics.update(self._model_opt(model_loss))
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        # dyn and rep loss have the same value
        metrics["dyn_rep_loss"] = to_np(dyn_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
        detached_post = {k1: {k2: v2.detach() for k2, v2 in v1.items()} for k1, v1 in posts}
        return detached_post, sample_data, metrics

    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        obs["reward"] = torch.Tensor(obs["reward"]).unsqueeze(-1)
        obs["target"] = torch.Tensor(obs["target"]).type(torch.IntTensor)
        obs["prev_target"] = torch.Tensor(obs["prev_target"]).type(torch.IntTensor)
        obs["front"] = torch.Tensor(obs["front"]).type(torch.int64)
        if "discount" in obs:
            obs["discount"] *= self._config.find_discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        if "is_terminal" in obs:
            # this label is necessary to train cont_head
            obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        else:
            raise ValueError('"is_terminal" was not found in observation.')
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    # Get the first 6 segment of this batch, for the first 5 frames, observe
    # for the rest of 64 - 5 frames, imagine
    # Top is ground truth, middle is restored and last is error
    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["image"](self.dynamics.get_feat(states)).mode()[:6]
        # reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        openl = self.heads["image"](self.dynamics.get_feat(prior)).mode()
        # reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6] + 0.5
        model = model + 0.5
        error = (model - truth + 1.0) / 2.0
        batches = torch.cat([truth, model, error], 2) * 255
        return torch.cat([b for b in batches], 2).type(torch.IntTensor)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._reward = reward
        self._device = config.device
        self.a2c_navigate = networks.A2CHead(
            self._config.dyn_stoch * self._config.dyn_discrete,
            config.dyn_deter,  # pytorch version
            config.num_actions,
            config.navigate_a2c_layers,
            config.units,
            embed_dim=config.embed_dim,
            unimix_ratio=config.action_unimix_ratio
        )
        self.a2c_explore = networks.A2CHead(
            self._config.dyn_stoch * self._config.dyn_discrete,
            config.dyn_deter,  # pytorch version
            config.num_actions,
            config.explore_a2c_layers,
            config.units,
            embed_dim=config.embed_dim,
            unimix_ratio=config.action_unimix_ratio,
        )
        self.a2c_combat = networks.A2CHead(
            self._config.dyn_stoch * self._config.dyn_discrete,
            config.dyn_deter,  # pytorch version
            config.num_actions,
            config.combat_a2c_layers,
            config.units,
            embed_dim=config.embed_dim,
            unimix_ratio=config.action_unimix_ratio,
        )
        kw = dict(opt=config.opt, use_amp=self._use_amp, wd=0, sub={"navigate/a2c": self.a2c_navigate,
                                                                    "explore/a2c": self.a2c_explore,
                                                                    "combat/a2c": self.a2c_combat})
        self._a2c_opt = tools.Optimizer(
            "a2c",
            [{'params': list(self.a2c_navigate.parameters()) + list(self.a2c_explore.parameters()) + list(self.a2c_combat.parameters()),
              'lr': config.ac_lr, 'weight_decay': config.A2C_weight_decay}],
            config.ac_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        posts,
        data_collection
    ):
        metrics = {}
        threshold = torch.tensor(self._config.a2c_regularize_threshold).to(self._config.device)
        coeff = torch.tensor(self._config.regularization).to(self._config.device)
        iter = [("navigate", "navigate_target", self.a2c_navigate, self._config.navigate_imag_horizon),
                ("explore", "navigate_target", self.a2c_explore, self._config.explore_imag_horizon),
                ("combat", "combat_target", self.a2c_combat, self._config.combat_imag_horizon)]
        total_loss = None
        for prefix, target_name, a2c_head, imag_horizon in iter:
            head_name = "{}/reward".format(prefix)
            post = posts[prefix]
            data = data_collection[prefix]
            flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
            with tools.RequiresGrad(a2c_head):
                with torch.cuda.amp.autocast(self._use_amp):
                    target_array = flatten(data[target_name].to(self._device))
                    imag_stoch, imag_deter, imag_state, imag_action, means, policy_params = self._imagine(
                        post, imag_horizon, target_array, a2c_head
                    )
                    value = tools.TwoHotDistSymlog(logits=means, device=self._device)
                    target_array_expanded = target_array.expand(imag_stoch.shape[0], target_array.shape[0])
                    reward = self._world_model.heads[head_name](imag_stoch, imag_deter, target_array_expanded).mode()
                    policy = tools.OneHotDist(policy_params, unimix_ratio=self._config.action_unimix_ratio)
                    actor_ent = policy.entropy()
                    state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                    # this target is not scaled
                    value_mode = value.mode().detach()
                    target, weights = self._compute_target(
                        imag_state, reward, value_mode
                    )
                    value = tools.TwoHotDistSymlog(logits=means[:-1], device=self._device)
                    value_mode = value.mode().detach()
                    actor_loss, mets = self._compute_actor_loss(
                        imag_action,
                        target,
                        actor_ent,
                        state_ent,
                        weights,
                        policy,
                        value_mode,
                        prefix
                    )
                    policy_abs = policy_params.abs()
                    policy_abs[policy_abs <= threshold] = 0
                    policy_abs[policy_abs > 0] -= threshold
                    policy_abs = policy_abs * weights
                    actor_suppressor = policy_abs[policy_abs > 0].mean() * coeff
                    if not actor_suppressor.isnan().any():
                        actor_loss += actor_suppressor
                    metrics.update(mets)
                    # (time, batch, 1), (time, batch, 1) -> (time, batch)
                    value_loss = -value.log_prob(target.detach())
                    # (time, batch, 1), (time, batch, 1) -> (1,)
                    value_abs = means[:-1].abs()
                    value_abs[value_abs <= threshold] = 0
                    value_abs[value_abs > 0] -= threshold
                    value_abs = value_abs * weights[:-1]
                    value_suppressor = value_abs[value_abs > 0].mean() * coeff
                    value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
                    if not value_suppressor.isnan().any():
                        value_loss += value_suppressor

            metrics.update(tools.tensorstats(policy_params, prefix + "/action_logits"))
            metrics.update(tools.tensorstats(means, prefix + "/value_logits"))
            metrics.update(tools.tensorstats(value.mode(), prefix + "/value"))
            metrics.update(tools.tensorstats(target, prefix + "/target"))
            metrics.update(tools.tensorstats(reward, prefix + "/imag_reward"))
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), prefix + "/imag_action"
                )
            )
            metrics[prefix + "/actor_ent"] = to_np(torch.mean(actor_ent))
            if total_loss is None:
                total_loss = actor_loss + value_loss
            else:
                total_loss += actor_loss + value_loss
            metrics[prefix + "/value_loss"] = value_loss.detach().cpu().numpy()
            metrics[prefix + "/actor_loss"] = actor_loss.detach().cpu().numpy()
            metrics[prefix + "/value_suppressor"] = value_suppressor.detach().cpu().numpy()
            metrics[prefix + "/actor_suppressor"] = actor_suppressor.detach().cpu().numpy()
        with tools.RequiresGrad([self.a2c_navigate, self.a2c_explore, self.a2c_combat]):
            metrics.update(self._a2c_opt(total_loss))
        return imag_stoch, imag_deter, imag_state, imag_action, weights, metrics

    def _imagine(self, start, horizon, target_array, a2c_head):
        dynamics = self._world_model.dynamics
        # start['deter'] (18, 64, 3072) -> (1152, 3072)
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _, _, _, _ = prev
            stoch, deter = dynamics.get_sep(state)
            stoch, deter = stoch.detach(), deter.detach()
            mean, policy_param = a2c_head(stoch, deter, target_array)
            policy = tools.OneHotDist(policy_param, unimix_ratio=self._config.action_unimix_ratio)
            action = policy.sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, stoch, deter, action, mean, policy_param

        # deters (15, 1152, 3072)
        succ, stoches, deters, actions, means, policy_params = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None, None, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return stoches, deters, states, actions, means, policy_params

    def lambda_return(self, reward, value, discount, l):
        # (15, 1152, 1)
        acc = value[-1]
        acc = acc.unsqueeze(0)
        # outputs is originally (1, 1152, 1)
        for i in range(len(value) - 2, -1, -1):
            curr = acc[-1] * l + value[i + 1] * (1 - l)
            curr = reward[i] + discount[i] * curr
            acc = torch.cat([acc, curr.unsqueeze(0)], dim=0)
        return acc[1:].flip(0)


    def _compute_target(
        self, imag_state, reward, value
    ):
        # reward r1 - r15 (r1 is the reward after taking the first action in the start state)
        # discount g1 - g15 (g1 is the futured value after r1)
        # value v1 - v15 (v1 is the value of the start state)
        inp = self._world_model.dynamics.get_feat(imag_state)
        discount = self._config.find_discount * self._world_model.heads["cont"](inp).mean.detach()
        # value(15, 960, ch)
        # action(15, 960, ch)
        # discount(15, 960, ch)
        target = self.lambda_return(reward, value, discount, self._config.discount_lambda)
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights

    def _compute_actor_loss(
        self,
        imag_action,
        target,
        actor_ent,
        state_ent,
        weights,
        policy,
        value,
        prefix
    ):
        metrics = {}
        # Q-val for actor is not transformed using symlog
        actor_target = (
            policy.log_prob(imag_action)[:-1][:, :, None]
            * (target - value).detach()
        )
        actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
        actor_target += actor_entropy
        metrics["{}/actor_entropy".format(prefix)] = to_np(torch.mean(actor_entropy))
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["{}/actor_state_entropy".format(prefix)] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics