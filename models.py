import copy

import torch
from torch import nn
import networks
import tools
from envs.crafter import targets
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
    def __init__(self, step, config, navigate_dataset, explore_dataset):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.navigate_dataset = navigate_dataset
        self.explore_dataset = explore_dataset

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
        self.heads["where"] = networks.DenseHead(
            feat_size,  # pytorch version
            (len(targets) * 4,),
            config.where_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )
        self.heads["explore_reward"] = networks.EmbeddedDenseHead(
            self._config.dyn_stoch * self._config.dyn_discrete,
            config.dyn_deter,
            config.explore_reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
        )
        self.heads["navigate_reward"] = networks.EmbeddedDenseHead(
            self._config.dyn_stoch * self._config.dyn_discrete,
            config.dyn_deter,
            config.navigate_reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
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
            sub={"cont": self.heads["cont"], "image": self.heads["image"], "encoder": self.encoder,
                 "rssm": self.dynamics, "explore_reward": self.heads["explore_reward"], "navigate_reward": self.heads["navigate_reward"],
                 "where": self.heads["where"]}
        )
        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale)

    def _train(self):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        conditional_metrics = {}
        metrics = {}
        threshold = torch.tensor(self._config.regularize_threshold).to("cuda")
        coeff = torch.tensor(self._config.regularization).to("cuda")
        target_dist = np.array(copy.copy(self.navigate_dataset.aggregate_sizes))
        target_dist = target_dist / np.sum(target_dist)
        navigate_data = self.preprocess(self.navigate_dataset.sample(target_dist))
        explore_data = self.preprocess(self.explore_dataset.sample(target_dist))
        data = {k: torch.cat([v, explore_data[v]], dim=0) for k, v in navigate_data.items() if k in explore_data}

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                navigate_embed = self.encoder(navigate_data)
                explore_embed = self.encoder(explore_data)
                navigate_post, navigate_prior = self.dynamics.observe(
                    navigate_embed, navigate_data["action"], navigate_data["is_first"]
                )
                explore_post, explore_prior = self.dynamics.observe(
                    explore_embed, explore_data["action"], explore_data["is_first"]
                )
                prior = {k: torch.cat([v, explore_prior[v]], dim=0) for k, v in navigate_prior.items() if k in explore_prior}
                post = {k: torch.cat([v, explore_post[v]], dim=0) for k, v in navigate_post.items() if k in explore_post}
                kl_free = tools.schedule(self._config.kl_free, self._step)
                dyn_scale = tools.schedule(self._config.dyn_scale, self._step)
                rep_scale = tools.schedule(self._config.rep_scale, self._step)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                losses = {}
                likes = {}
                for name, head in self.heads.items():
                    if "reward" not in name:
                        feat = self.dynamics.get_feat(post)
                        pred = head(feat)
                        like = pred.log_prob(data[name])

                        likes[name] = like
                        losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)

                    if "reward" in name:
                        (stoch, deter) = self.dynamics.get_sep(explore_post)
                        pred = head(stoch, deter, data["target"])
                        like = pred.log_prob(data[name])
                        likes[name] = like
                        losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                        reward_logits = pred.logits.abs()
                        reward_suppressor = (reward_logits[reward_logits > threshold] - threshold).mean() * coeff
                        if not reward_suppressor.isnan().any():
                            losses[name] += reward_suppressor
                        metrics.update(tools.tensorstats(pred.logits, name + "_logits"))

                model_loss = sum(losses.values()) + kl_loss
                metrics.update(self._model_opt(model_loss))
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["reward_suppressor"] = to_np(reward_suppressor)
        metrics["kl"] = to_np(torch.mean(kl_value))
        metrics = {**metrics, **conditional_metrics}
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
        navigate_post = {k: v.detach() for k, v in navigate_post.items()}
        explore_post = {k: v.detach() for k, v in explore_post.items()}
        return navigate_post, explore_post, navigate_data, explore_data, metrics

    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        obs["reward"] = torch.Tensor(obs["reward"]).unsqueeze(-1)
        obs["target"] = torch.Tensor(obs["target"]).type(torch.IntTensor)
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

        return (torch.cat([truth, model, error], 2) * 255).type(torch.IntTensor)


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
            unimix_ratio=config.action_unimix_ratio,
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
        kw = dict(opt=config.opt, use_amp=self._use_amp, wd=0, sub={"a2c": self.a2c_navigate})
        self._a2c_opt = tools.Optimizer(
            "a2c",
            [{'params': self.a2c_navigate.parameters(), 'lr': config.ac_lr, 'weight_decay': config.A2C_weight_decay}],
            config.ac_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        navigate_post,
        explore_post,
        navigate_data,
        explore_data
    ):
        metrics = {}
        threshold = torch.tensor(self._config.a2c_regularize_threshold).to("cuda")
        coeff = torch.tensor(self._config.regularization).to("cuda")
        iter = [("navigate", "navigate_reward", navigate_post, navigate_data), ("explore", "explore_reward", explore_post, explore_data)]
        for prefix, head_name, post, data in iter:
            with tools.RequiresGrad(self):
                with torch.cuda.amp.autocast(self._use_amp):
                    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                    target_array = torch.from_numpy(flatten(data["target"])).to(self._device)
                    imag_stoch, imag_deter, imag_state, imag_action, means, policy_params = self._imagine(
                        post, self._config.imag_horizon, target_array,
                    )
                    value = tools.TwoHotDistSymlog(logits=means)
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
                    value = tools.TwoHotDistSymlog(logits=means[:-1])
                    value_mode = value.mode().detach()
                    actor_loss, mets = self._compute_actor_loss(
                        imag_action,
                        target,
                        actor_ent,
                        state_ent,
                        weights,
                        policy,
                        value_mode
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

            metrics.update(tools.tensorstats(policy_params, prefix + "_action_logits"))
            metrics.update(tools.tensorstats(means, prefix + "_value_logits"))
            metrics.update(tools.tensorstats(value.mode(), prefix + "_value"))
            metrics.update(tools.tensorstats(target, prefix + "_target"))
            metrics.update(tools.tensorstats(reward, prefix + "_imag_reward"))
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), prefix + "_imag_action"
                )
            )
            metrics[prefix + "_actor_ent"] = to_np(torch.mean(actor_ent))
            with tools.RequiresGrad(self):
                metrics.update(self._a2c_opt(actor_loss + value_loss))
            metrics[prefix + "_value_loss"] = value_loss.detach().cpu().numpy()
            metrics[prefix + "_actor_loss"] = actor_loss.detach().cpu().numpy()
            metrics[prefix + "_value_suppressor"] = value_suppressor.detach().cpu().numpy()
            metrics[prefix + "_actor_suppressor"] = actor_suppressor.detach().cpu().numpy()
        return imag_stoch, imag_deter, imag_state, imag_action, weights, metrics

    def _imagine(self, start, horizon, target_array):
        dynamics = self._world_model.dynamics
        # start['deter'] (18, 64, 3072) -> (1152, 3072)
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _, _, _, _ = prev
            stoch, deter = dynamics.get_sep(state)
            stoch, deter = stoch.detach(), deter.detach()
            mean, policy_param = self.a2c_navigate(stoch, deter, target_array)
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
        value
    ):
        metrics = {}
        # Q-val for actor is not transformed using symlog
        actor_target = (
            policy.log_prob(imag_action)[:-1][:, :, None]
            * (target - value).detach()
        )
        actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
        actor_target += actor_entropy
        metrics["actor_entropy"] = to_np(torch.mean(actor_entropy))
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics