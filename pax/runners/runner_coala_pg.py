"""Runner for COALA-PG (Meulemans et al., ICLR 2025; arXiv:2410.18636).

This realises the paper's *batched co-player shaping POMDP*:

  * ``num_envs``        = B, the lockstep batch the co-player learns from
  * ``num_outer_steps`` = M, inner episodes per meta-trajectory
  * ``num_inner_steps`` = T, steps per inner episode
  * ``num_opps``        = independent replicas of the whole construction

The co-player is re-initialised at the start of every meta-trajectory and
updates its parameters at each inner-episode boundary using all B trajectories
(this is already how Pax batches agent 2). The ego-agent keeps a single GRU
hidden state across all M episodes, so it conditions on the co-player's whole
learning trace.

The one structural difference from `runner_marl.RLRunner`: agent 1's trajectory
is handed to `update` as ``[M*T, num_opps, num_envs, ...]`` instead of being
flattened by `reduce_outer_traj`. Collapsing ``num_opps`` into the batch would
make the COALA cross-batch average run over *independent* co-players, which is
wrong — the coupling exists only within one co-player's minibatch.

Scope note: the co-player population here is homogeneous (every co-player is a
naive learner), which is the setting that matches a "shaper vs naive learner"
baseline. The paper's mixed naive/learning-aware co-player population
(``p_naive``) is not implemented.
"""

import os
import time
from datetime import datetime
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.utils import MemoryState, TrainingState, save
from pax.watchers import cg_visitation, ipd_visitation

MAX_WANDB_CALLS = 1000


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


def to_long_trajectory(traj: Sample) -> Sample:
    """``[M, T, num_opps, num_envs, ...]`` -> ``[M*T, num_opps, num_envs, ...]``.

    Deliberately keeps ``num_opps`` and ``num_envs`` separate; contrast with
    `runner_marl.reduce_outer_traj`, which merges them.
    """
    return jax.tree_util.tree_map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), traj
    )


class CoalaPGRunner:
    """Trains a COALA-PG shaper against a learning co-player."""

    def __init__(self, agents, env, save_dir, args):
        self.args = args
        self.env = env
        self.save_dir = save_dir
        self.random_key = jax.random.PRNGKey(args.seed)
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        self.train_steps = 0
        self.train_episodes = 0
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)

        # ---- VMAP env over num_envs, then num_opps ----
        env.batch_reset = jax.vmap(env.reset, (0, None), 0)
        env.batch_step = jax.vmap(env.step, (0, 0, 0, None), 0)
        env.batch_reset = jax.jit(jax.vmap(env.batch_reset, (0, None), 0))
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step, (0, 0, 0, None), 0)
        )

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))

        agent1, agent2 = agents
        num_outer_steps = args.num_outer_steps
        coala_objective = args.get("coala_objective", "selfish")
        if coala_objective not in ("selfish", "welfare", "constrained_welfare"):
            raise ValueError(
                "coala_objective must be 'selfish', 'welfare', or "
                "'constrained_welfare'. "
                f"Got {coala_objective}."
            )
        if coala_objective == "constrained_welfare":
            self.mu1 = float(args.welfare.mu1)
            self.mu2 = float(args.welfare.mu2)
            self.dual_lr = float(args.welfare.dual_lr)
            self.v_ref_shaper = float(args.welfare.v_ref_shaper)
            self.v_ref_opponent = float(args.welfare.v_ref_opponent)
            self.rho1 = float(args.welfare.rho1)
            self.rho2 = float(args.welfare.rho2)
            self.rho_multiplier = float(args.welfare.rho_schedule)
            self.rho_patience = int(args.welfare.rho_patience)
            self.rho_max = float(args.welfare.rho_max)
            self.violation_counter_1 = 0
            self.violation_counter_2 = 0
        else:
            self.mu1 = 0.0
            self.mu2 = 0.0
            self.dual_lr = 0.0
            self.v_ref_shaper = 0.0
            self.v_ref_opponent = 0.0
            self.rho1 = 0.0
            self.rho2 = 0.0
            self.rho_multiplier = 1.0
            self.rho_patience = 0
            self.rho_max = 0.0
            self.violation_counter_1 = 0
            self.violation_counter_2 = 0

        # ---- agent 1 (COALA-PG shaper): batched over num_opps ----
        agent1.batch_init = jax.vmap(
            agent1.make_initial_state, (None, 0), (None, 0)
        )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )
        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # ---- agent 2 (co-player): batched over num_opps ----
        agent2.batch_init = jax.vmap(agent2.make_initial_state, (0, None), 0)
        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy))
        agent2.batch_reset = jax.jit(
            jax.vmap(agent2.reset_memory, (0, None), 0), static_argnums=1
        )
        # in_axes=(1, 0, 0, 0): the trajectory's num_opps axis is 1, so each
        # co-player's update consumes all num_envs (= B) trajectories.
        agent2.batch_update = jax.jit(jax.vmap(agent2.update, (1, 0, 0, 0), 0))

        init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
        agent1._state, agent1._mem = agent1.batch_init(
            agent1._state.random_key, init_hidden
        )

        a2_rng = jax.random.split(agent2._state.random_key, args.num_opps)
        agent2._state, agent2._mem = agent2.batch_init(
            a2_rng, jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))
        )

        def _inner_rollout(carry, unused):
            """One step of an inner episode."""
            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = carry

            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            rngs = rngs[:, :, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state, obs1, a1_mem
            )
            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state, obs2, a2_mem
            )
            (next_obs1, next_obs2), env_state, rewards, done, info = (
                env.batch_step(env_rng, env_state, (a1, a2), env_params)
            )

            traj1 = Sample(
                obs1,
                a1,
                rewards[0],
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done,
                a1_mem.hidden,
            )
            traj2 = Sample(
                obs2,
                a2,
                rewards[1],
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done,
                a2_mem.hidden,
            )
            return (
                rngs,
                next_obs1,
                next_obs2,
                rewards[0],
                rewards[1],
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
                env_params,
            ), (traj1, traj2)

        def _outer_rollout(carry, unused):
            """One inner episode, followed by the co-player's learning step."""
            vals, trajectories = jax.lax.scan(
                _inner_rollout, carry, None, length=self.args.num_inner_steps
            )
            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = vals

            # Co-player learns from its minibatch of all num_envs trajectories.
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1], obs2, a2_state, a2_mem
            )
            # NOTE: a1_mem is intentionally NOT reset here. Carrying the GRU
            # state across the episode boundary is what gives the shaper its
            # long-context view of the co-player's learning dynamics.
            return (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ), (*trajectories, a2_metrics)

        def _rollout(
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _a2_state: TrainingState,
            _a2_mem: MemoryState,
            _env_params: Any,
            _objective_state: jnp.ndarray,
        ):
            """One meta-trajectory: M inner episodes against a fresh co-player."""
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.batch_reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs)),
                jnp.zeros((args.num_opps, args.num_envs)),
            ]

            # Fresh shaper memory, and a freshly initialised co-player: the
            # shaper must shape learning from scratch each meta-trajectory.
            _a1_mem = agent1.batch_reset(_a1_mem, False)
            a2_rng = jax.random.split(_rng_run, args.num_opps)
            _a2_state, _a2_mem = agent2.batch_init(a2_rng, _a2_mem.hidden)

            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    *obs,
                    *rewards,
                    _a1_state,
                    _a1_mem,
                    _a2_state,
                    _a2_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=num_outer_steps,
            )
            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                _env_params,
            ) = vals
            traj_1, traj_2, a2_metrics = stack

            # Per-episode returns: [M, num_opps, num_envs] -> [M]
            ep_rewards_1 = traj_1.rewards.sum(axis=1).mean(axis=(1, 2))
            ep_rewards_2 = traj_2.rewards.sum(axis=1).mean(axis=(1, 2))

            # COALA-PG update over the whole meta-trajectory. The standard
            # paper baseline is selfish (`r_1`). Welfare modes change only the
            # optimizer objective reward stream; evaluation/logging keeps the
            # true individual rewards. The constrained welfare mode mirrors the
            # augmented Lagrangian used by `runner_welfare_evo`:
            #
            #   W + mu*s - rho/2 * max(0, -s)^2
            #
            # For a policy-gradient update, the return-dependent derivative is
            # an episode-wide reward weight: 1 + mu + rho * max(0, -s).
            objective_weights = _objective_state[:2]
            if coala_objective == "constrained_welfare":
                slack_1 = ep_rewards_1.mean() - _objective_state[6]
                slack_2 = ep_rewards_2.mean() - _objective_state[7]
                objective_weights = jnp.asarray(
                    [
                        1.0
                        + _objective_state[2]
                        + _objective_state[4] * jnp.maximum(0.0, -slack_1),
                        1.0
                        + _objective_state[3]
                        + _objective_state[5] * jnp.maximum(0.0, -slack_2),
                    ]
                )
            objective_rewards = (
                objective_weights[0] * traj_1.rewards
                + objective_weights[1] * traj_2.rewards
            )
            objective_traj_1 = traj_1._replace(rewards=objective_rewards)
            long_traj_1 = to_long_trajectory(objective_traj_1)
            a1_state, a1_mem, a1_metrics = agent1.update(
                long_traj_1, obs1, a1_state, a1_mem
            )
            a1_metrics["coala/objective_is_welfare"] = jnp.asarray(
                coala_objective in ("welfare", "constrained_welfare")
            )
            a1_metrics["coala/objective_weight_player_1"] = objective_weights[0]
            a1_metrics["coala/objective_weight_player_2"] = objective_weights[1]

            if args.env_id == "iterated_matrix_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        traj_1.observations, traj_1.actions, obs1
                    ),
                )
            else:
                env_stats = {}

            return (
                env_stats,
                ep_rewards_1,
                ep_rewards_2,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
                objective_weights,
            )

        self.rollout = jax.jit(_rollout)

    def _objective_state(self) -> jnp.ndarray:
        coala_objective = self.args.get("coala_objective", "selfish")
        if coala_objective == "selfish":
            base_weights = [1.0, 0.0]
        elif coala_objective == "welfare":
            base_weights = [1.0, 1.0]
        else:
            base_weights = [1.0 + self.mu1, 1.0 + self.mu2]
        return jnp.asarray(
            [
                base_weights[0],
                base_weights[1],
                self.mu1,
                self.mu2,
                self.rho1,
                self.rho2,
                self.v_ref_shaper,
                self.v_ref_opponent,
            ]
        )

    def run_loop(self, env_params, agents, num_iters, watchers):
        print("Training COALA-PG")
        print("-----------------------")
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        log_interval = int(max(num_iters / MAX_WANDB_CALLS, 5))
        print(f"Number of meta-trajectories (iterations): {num_iters}")
        print(f"Inner episodes per meta-trajectory (M): {self.args.num_outer_steps}")
        print(f"Inner episode length (T): {self.args.num_inner_steps}")
        print(f"Co-player batch (B = num_envs): {self.args.num_envs}")
        print(f"Independent co-players (num_opps): {self.args.num_opps}")
        print(f"Log interval: {log_interval}")

        for i in range(num_iters):
            rng, rng_run = jax.random.split(rng, 2)
            objective_state = self._objective_state()
            (
                env_stats,
                ep_rewards_1,
                ep_rewards_2,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
                objective_weights,
            ) = self.rollout(
                rng_run,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_params,
                objective_state,
            )
            mean_return_1 = float(ep_rewards_1.mean())
            mean_return_2 = float(ep_rewards_2.mean())
            slack_1 = mean_return_1 - self.v_ref_shaper
            slack_2 = mean_return_2 - self.v_ref_opponent

            if self.args.get("coala_objective", "selfish") == "constrained_welfare":
                self.mu1 = max(0.0, self.mu1 - self.dual_lr * slack_1)
                self.mu2 = max(0.0, self.mu2 - self.dual_lr * slack_2)

                if mean_return_1 < self.v_ref_shaper:
                    self.violation_counter_1 += 1
                else:
                    self.violation_counter_1 = 0
                if mean_return_2 < self.v_ref_opponent:
                    self.violation_counter_2 += 1
                else:
                    self.violation_counter_2 = 0

                if self.violation_counter_1 >= self.rho_patience:
                    self.rho1 = min(self.rho1 * self.rho_multiplier, self.rho_max)
                    self.violation_counter_1 = 0
                    print(
                        f"[Lagrangian] rho1 (shaper) increased to {self.rho1:.4f}"
                    )
                if self.violation_counter_2 >= self.rho_patience:
                    self.rho2 = min(self.rho2 * self.rho_multiplier, self.rho_max)
                    self.violation_counter_2 = 0
                    print(
                        f"[Lagrangian] rho2 (opponent) increased to {self.rho2:.4f}"
                    )
            next_objective_state = self._objective_state()

            if i % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"iteration_{i}")
                save(a1_state.params, log_savepath)
                if watchers:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {i} locally")

            if i % log_interval == 0:
                # First vs last inner episode is the shaping signal: it shows
                # what the co-player's learning was steered toward.
                first_1, last_1 = float(ep_rewards_1[0]), float(ep_rewards_1[-1])
                first_2, last_2 = float(ep_rewards_2[0]), float(ep_rewards_2[-1])
                print(f"Iteration {i}")
                print(
                    f"  episode 1   : shaper {first_1:.4f} | co-player {first_2:.4f}"
                )
                print(
                    f"  episode {self.args.num_outer_steps:<3} : "
                    f"shaper {last_1:.4f} | co-player {last_2:.4f}"
                )
                print(
                    f"  meta-mean   : shaper {mean_return_1:.4f} | "
                    f"co-player {mean_return_2:.4f} | "
                    f"welfare {mean_return_1 + mean_return_2:.4f}"
                )
                if self.args.get("coala_objective", "selfish") == "constrained_welfare":
                    print(
                        f"  constraints : mu_shaper {self.mu1:.4f} | "
                        f"mu_co-player {self.mu2:.4f} | "
                        f"slack_shaper {slack_1:.4f} | "
                        f"slack_co-player {slack_2:.4f} | "
                        f"rho_shaper {self.rho1:.4f} | rho_co-player {self.rho2:.4f}"
                    )
                for stat, val in env_stats.items():
                    print(f"  {stat}: {float(val)}")
                print()

                if watchers:
                    flat_a1 = jax.tree_util.tree_map(jnp.mean, a1_metrics)
                    agent1._logger.metrics = (
                        agent1._logger.metrics | flat_a1
                    )
                    flat_a2 = jax.tree_util.tree_map(jnp.mean, a2_metrics)
                    agent2._logger.metrics = (
                        agent2._logger.metrics | flat_a2
                    )
                    for watcher, agent in zip(watchers, agents):
                        watcher(agent)

                    wandb.log(
                        {
                            "train_iteration": i,
                            "train/reward_per_episode/player_1": float(
                                ep_rewards_1.mean()
                            ),
                            "train/reward_per_episode/player_2": float(
                                ep_rewards_2.mean()
                            ),
                            "train/welfare": float(
                                ep_rewards_1.mean() + ep_rewards_2.mean()
                            ),
                            "train/first_episode/player_1": first_1,
                            "train/first_episode/player_2": first_2,
                            "train/final_episode/player_1": last_1,
                            "train/final_episode/player_2": last_2,
                            # Shaping signal: how much the co-player's return
                            # improved from the first to the last inner episode.
                            "train/shaping_delta/player_2": last_2 - first_2,
                            "train/lagrangian/mu_shaper": self.mu1,
                            "train/lagrangian/mu_opponent": self.mu2,
                            "train/lagrangian/v_ref_shaper": self.v_ref_shaper,
                            "train/lagrangian/v_ref_opponent": self.v_ref_opponent,
                            "train/lagrangian/slack_shaper": slack_1,
                            "train/lagrangian/slack_opponent": slack_2,
                            "train/lagrangian/rho_shaper": self.rho1,
                            "train/lagrangian/rho_opponent": self.rho2,
                            "train/lagrangian/violation_counter_shaper": (
                                self.violation_counter_1
                            ),
                            "train/lagrangian/violation_counter_opponent": (
                                self.violation_counter_2
                            ),
                            "train/lagrangian/objective_weight_used_player_1": float(
                                objective_weights[0]
                            ),
                            "train/lagrangian/objective_weight_used_player_2": float(
                                objective_weights[1]
                            ),
                            "train/lagrangian/objective_weight_next_player_1": float(
                                next_objective_state[0]
                            ),
                            "train/lagrangian/objective_weight_next_player_2": float(
                                next_objective_state[1]
                            ),
                        }
                        | {k: float(v) for k, v in env_stats.items()}
                        # The shared `ppo_memory_log` watcher logs nothing for
                        # this agent, so surface the COALA-PG diagnostics
                        # (advantage magnitudes, losses, grad norms) here.
                        | {
                            f"train/shaper/{k}": float(v)
                            for k, v in flat_a1.items()
                        },
                    )

        agents[0]._state = a1_state
        agents[1]._state = a2_state
        return agents
