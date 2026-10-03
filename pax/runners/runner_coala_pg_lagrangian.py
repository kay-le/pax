"""Runner for COALA-PG on a constrained-welfare Lagrangian.

Solves

    max_theta  W(theta)        s.t.   R_s(theta) >= tau

where ``W = R_s + R_o`` is welfare over the meta-episode and ``R_s`` is the
shaper's own return. Because the constraint is itself an expected return, the
Lagrangian

    L(theta, lam) = W(theta) + lam * (R_s(theta) - tau)

has both terms as expectations over the *same* trajectories, so its gradient is
just the COALA-PG gradient of a reshaped reward:

    r_mix = (r_s + r_o) + lam * r_s = (1 + lam) * r_s + r_o.

Nothing in the learning-aware estimator changes. Relative to
`runner_coala_pg.CoalaPGRunner`, exactly three things differ:

  1. the trajectory carries two reward streams (welfare and constraint),
  2. the baseline comes from two critic heads combined as V_W + lam * V_c,
  3. a lam controller runs after each meta-trajectory.

The rollout structure, the co-player, and `coala_advantages` are untouched,
which is what lets a selfish / welfare / constrained comparison be attributed
to the objective rather than to the optimizer or the architecture.

Reference for the dual update: Stooke, Achiam & Abbeel, "Responsive Safety in
Reinforcement Learning by PID Lagrangian Methods", ICML 2020. Setting
``welfare.kp = welfare.kd = 0`` recovers RCPO (Tessler et al., 2019) dual
ascent exactly, so the dual-ascent baseline needs no separate runner.

Two modelling choices that matter more than the controller gains:

  * THE CONSTRAINT WINDOW. A floor on the shaper's average return over the
    whole meta-episode can be satisfied by exploiting the co-player early and
    being exploited late, after it has converged -- the opposite of "shaping
    established cooperation". `welfare.constraint_window` restricts the
    constraint to the last K inner episodes, making tau a statement about
    where the co-player ends up.
  * THE SCALE OF tau. The constraint return is normalised to a PER-INNER-
    EPISODE scale, so tau can be read straight off the payoff matrix (e.g. the
    mutual-defection payoff, which makes the constraint exactly the
    individual-rationality condition and "self-sacrifice" the well-defined
    event R_s < tau). If the ES/Shaper runs define their floor on total
    meta-episode return instead, rescale one of them so the constraint means
    the same thing on both optimizers.
"""

import os
import time
from datetime import datetime
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.runners.pid_lagrangian import PIDLagrangian
from pax.utils import MemoryState, TrainingState, save
from pax.watchers import cg_visitation, ipd_visitation

MAX_WANDB_CALLS = 1000


class Sample(NamedTuple):
    """A batch of data.

    `rewards` is the welfare stream; `cost_rewards` is the constraint stream
    (the shaper's own reward, masked to the constraint window and rescaled).
    `behavior_values` carries both critic heads on a trailing axis of size 2.
    """

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    cost_rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


def to_long_trajectory(traj: Sample) -> Sample:
    """``[M, T, num_opps, num_envs, ...]`` -> ``[M*T, num_opps, num_envs, ...]``.

    Keeps ``num_opps`` and ``num_envs`` separate: collapsing them would make
    the COALA cross-batch average run over *independent* co-players, which is
    wrong -- the coupling exists only within one co-player's minibatch.
    """
    return jax.tree_util.tree_map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), traj
    )


def constraint_window_mask(
    num_outer_steps: int, window: int, dtype=jnp.float32
) -> jnp.ndarray:
    """Weights over the M inner episodes selecting the constraint window.

    Returns ``[M]``, zero outside the last ``window`` episodes and
    ``M / window`` inside it. The scale is chosen so that

        sum_m mask[m] * ep_return[m] / M  ==  mean of ep_return over the window

    i.e. the constraint stream integrates to the windowed per-episode mean
    multiplied by M -- the same scale as the unwindowed self-return. With
    ``window >= M`` the mask is all ones and the reward stream is exactly
    ``r_s``, so ``r_mix = (1 + lam) * r_s + r_o`` with no hidden factor.
    """
    if window <= 0 or window >= num_outer_steps:
        return jnp.ones((num_outer_steps,), dtype=dtype)
    idx = jnp.arange(num_outer_steps)
    inside = idx >= (num_outer_steps - window)
    return jnp.where(inside, num_outer_steps / float(window), 0.0).astype(dtype)


class CoalaPGLagrangianRunner:
    """Trains a Lagrangian-constrained COALA-PG shaper against a learner."""

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

        w = args.welfare
        self.tau = float(w.tau)
        self.constraint_window = int(w.get("constraint_window", 0))
        self.controller = PIDLagrangian(
            tau=self.tau,
            kp=float(w.get("kp", 0.0)),
            ki=float(w.get("ki", 0.01)),
            kd=float(w.get("kd", 0.0)),
            lam_init=float(w.get("lam_init", 0.5)),
            lam_max=(
                None
                if w.get("lam_max", None) in (None, "null", "")
                else float(w.get("lam_max"))
            ),
            ema_beta=float(w.get("ema_beta", 0.0)),
        )
        # A fixed multiplier turns this into the static weighted-welfare
        # ablation: r_mix = (1 + lam) * r_s + r_o with lam never updated. Worth
        # running, because a reviewer will ask whether an adaptive multiplier
        # beats a tuned constant.
        self.freeze_lam = bool(w.get("freeze_lam", False))
        self.lam = self.controller.lam

        # ---- VMAP env over num_envs, then num_opps ----
        env.batch_reset = jax.vmap(env.reset, (0, None), 0)
        env.batch_step = jax.vmap(env.step, (0, 0, 0, None), 0)
        env.batch_reset = jax.jit(jax.vmap(env.batch_reset, (0, None), 0))
        env.batch_step = jax.jit(jax.vmap(env.batch_step, (0, 0, 0, None), 0))

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))

        agent1, agent2 = agents
        num_outer_steps = args.num_outer_steps
        window_mask = constraint_window_mask(
            num_outer_steps, self.constraint_window
        )
        self._window_mask = window_mask

        # ---- agent 1 (Lagrangian COALA-PG shaper): batched over num_opps ----
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

            # Welfare stream now; the constraint stream needs the episode index
            # for window masking, so it is built in `_outer_rollout`.
            traj1 = Sample(
                obs1,
                a1,
                rewards[0] + rewards[1],
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
                rewards[1],
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done,
                a2_mem.hidden,
            )
            # Keep the raw per-player rewards for logging and the controller.
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
            ), (traj1, traj2, rewards[0], rewards[1])

        def _outer_rollout(carry, unused):
            """One inner episode, followed by the co-player's learning step."""
            vals, (traj1, traj2, raw_r1, raw_r2) = jax.lax.scan(
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
                traj2, obs2, a2_state, a2_mem
            )
            # a1_mem is intentionally NOT reset: carrying the GRU state across
            # the episode boundary is what gives the shaper its long-context
            # view of the co-player's learning dynamics.
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
            ), (traj1, traj2, a2_metrics, raw_r1, raw_r2)

        def _rollout(
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _a2_state: TrainingState,
            _a2_mem: MemoryState,
            _env_params: Any,
            _lam: jnp.ndarray,
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
            traj_1, traj_2, a2_metrics, raw_r1, raw_r2 = stack

            # Per-episode returns: [M, T, num_opps, num_envs] -> [M]
            ep_rewards_1 = raw_r1.sum(axis=1).mean(axis=(1, 2))
            ep_rewards_2 = raw_r2.sum(axis=1).mean(axis=(1, 2))

            # Apply the constraint window to the cost stream. window_mask is
            # [M]; the stream is [M, T, num_opps, num_envs].
            mask = window_mask.reshape((num_outer_steps, 1, 1, 1))
            traj_1 = traj_1._replace(cost_rewards=traj_1.cost_rewards * mask)

            # The quantity the controller regulates: the shaper's mean
            # per-inner-episode return over the constraint window, on the same
            # scale as tau.
            constrained_return = jnp.sum(
                window_mask * ep_rewards_1
            ) / float(num_outer_steps)

            long_traj_1 = to_long_trajectory(traj_1)
            a1_state, a1_mem, a1_metrics = agent1.update(
                long_traj_1, obs1, a1_state, a1_mem, _lam
            )

            if args.env_id == "iterated_matrix_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(traj_1.observations, traj_1.actions, obs1),
                )
            else:
                env_stats = {}

            return (
                env_stats,
                ep_rewards_1,
                ep_rewards_2,
                constrained_return,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            )

        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, num_iters, watchers):
        print("Training COALA-PG (Lagrangian constrained welfare)")
        print("-----------------------")
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        log_interval = int(max(num_iters / MAX_WANDB_CALLS, 5))
        M = self.args.num_outer_steps
        print(f"Number of meta-trajectories (iterations): {num_iters}")
        print(f"Inner episodes per meta-trajectory (M): {M}")
        print(f"Inner episode length (T): {self.args.num_inner_steps}")
        print(f"Co-player batch (B = num_envs): {self.args.num_envs}")
        print(f"Independent co-players (num_opps): {self.args.num_opps}")
        print(f"Log interval: {log_interval}")
        print("Constrained welfare (Lagrangian):")
        print(f"  objective: W + lam * (R_s - tau), tau = {self.tau}")
        window = (
            "whole meta-episode"
            if self.constraint_window <= 0 or self.constraint_window >= M
            else f"last {self.constraint_window} of {M} inner episodes"
        )
        print(f"  constraint window: {window}")
        c = self.controller
        print(
            f"  PID gains: kp={c.kp} ki={c.ki} kd={c.kd}"
            + ("   (kp=kd=0 => RCPO dual ascent)" if c.kp == 0 and c.kd == 0 else "")
        )
        print(
            f"  lam_init={c.lam} lam_max={c.lam_max} ema_beta={c.ema_beta}"
            + ("   [FROZEN: static weighted-welfare ablation]" if self.freeze_lam else "")
        )
        print(f"  tau is per inner episode, so read it off the payoff matrix.")

        for i in range(num_iters):
            rng, rng_run = jax.random.split(rng, 2)
            lam_used = self.lam
            (
                env_stats,
                ep_rewards_1,
                ep_rewards_2,
                constrained_return,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            ) = self.rollout(
                rng_run,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_params,
                jnp.asarray(lam_used, dtype=jnp.float32),
            )

            # ---- dual update, on the host, after the primal step ----
            r_constrained = float(constrained_return)
            if self.freeze_lam:
                self.controller.delta = self.tau - r_constrained
            else:
                self.lam = self.controller.update(r_constrained)

            if i % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"iteration_{i}")
                save(a1_state.params, log_savepath)
                if watchers:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {i} locally")

            if i % log_interval == 0:
                first_1, last_1 = float(ep_rewards_1[0]), float(ep_rewards_1[-1])
                first_2, last_2 = float(ep_rewards_2[0]), float(ep_rewards_2[-1])
                violation = self.tau - r_constrained
                print(f"Iteration {i}")
                print(
                    f"  episode 1   : shaper {first_1:.4f} | co-player {first_2:.4f}"
                )
                print(
                    f"  episode {M:<3} : shaper {last_1:.4f} | co-player {last_2:.4f}"
                )
                print(
                    f"  meta-mean   : shaper {float(ep_rewards_1.mean()):.4f} | "
                    f"co-player {float(ep_rewards_2.mean()):.4f} | "
                    f"welfare {float(ep_rewards_1.mean() + ep_rewards_2.mean()):.4f}"
                )
                # R_s is the windowed quantity the constraint is actually on;
                # it differs from meta-mean whenever a window is in use.
                print(
                    f"  constraint  : R_s {r_constrained:.4f} vs tau {self.tau:.4f}"
                    f" | violation {violation:+.4f}"
                    f" | {'VIOLATED' if violation > 0 else 'satisfied'}"
                )
                print(
                    f"  multiplier  : lam_used {lam_used:.4f} -> lam_next "
                    f"{self.lam:.4f}  (I {self.controller._integral:.4f} | "
                    f"P {self.controller.p_term:+.4f} | "
                    f"D {self.controller.d_term:+.4f})"
                )
                print(
                    f"  self weight : (1 + lam) = {1.0 + lam_used:.4f} on r_s,"
                    f" 1.0 on r_o"
                )
                for stat, val in env_stats.items():
                    print(f"  {stat}: {float(val)}")
                print()

                if watchers:
                    flat_a1 = jax.tree_util.tree_map(jnp.mean, a1_metrics)
                    agent1._logger.metrics = agent1._logger.metrics | flat_a1
                    flat_a2 = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                    )
                    agent2._logger.metrics = agent2._logger.metrics | flat_a2
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
                            "train/shaping_delta/player_2": last_2 - first_2,
                            # The constraint, the multiplier, and the effective
                            # reward weight it implies.
                            "train/lagrangian/constrained_return": r_constrained,
                            "train/lagrangian/lam_used": lam_used,
                            "train/lagrangian/self_weight": 1.0 + lam_used,
                            "train/lagrangian/feasible": float(violation <= 0),
                        }
                        | {
                            f"train/{k}": float(v)
                            for k, v in self.controller.metrics().items()
                        }
                        | {k: float(v) for k, v in env_stats.items()}
                        | {
                            f"train/shaper/{k}": float(v)
                            for k, v in flat_a1.items()
                        },
                    )

        agents[0]._state = a1_state
        agents[1]._state = a2_state
        return agents
