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

            # ---- PG-specific safeguards on the Lagrangian (see below) ----
            #
            # `runner_welfare_evo` can leave mu and the penalty uncapped because
            # OpenES + Adam is scale-invariant to a uniform rescaling of
            # fitness: ES has no critic. COALA-PG does, and the Lagrangian
            # enters as a *reward weight*, so an unbounded weight is an
            # unbounded reward scale. With gamma=1.0 over M*T steps that
            # de-calibrates the critic (see the horizon discussion in
            # `pax/agents/coala_pg/coala_pg.py`), the advantage becomes noise,
            # and the policy diffuses to uniform-random. Measured on IPD at
            # v_ref=(-15,-15): mu1 integrated to 50, rho1 pinned at rho_max=50,
            # giving w1 = 1 + 50 + 50*4.1 = 255 and a per-state cooperation
            # probability of 0.20 at every state (i.e. no policy at all).
            #
            # Three safeguards, all of which leave the unconstrained case
            # (mu=0, slack>=0 => w=(1,1)) bit-for-bit unchanged:
            self.mu_max = float(args.welfare.get("mu_max", 5.0))
            self.weight_max = float(args.welfare.get("weight_max", 10.0))
            self.weight_normalization = bool(
                args.welfare.get("weight_normalization", True)
            )
            self.slack_per_episode = bool(
                args.welfare.get("slack_per_episode", True)
            )
            if args.welfare.get("calibration", False):
                # ES implements this via `runner_welfare_evo.calibrate`; there
                # is no equivalent here, and silently ignoring the flag would
                # run with v_ref=0.0 instead of a calibrated reference.
                raise NotImplementedError(
                    "welfare.calibration is not implemented for runner=coala_pg. "
                    "Set calibration: False and supply v_ref_shaper / "
                    "v_ref_opponent explicitly."
                )
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
            self.mu_max = 0.0
            self.weight_max = 0.0
            self.weight_normalization = False
            self.slack_per_episode = False

        weight_normalization = self.weight_normalization
        slack_per_episode = self.slack_per_episode

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
            if coala_objective == "constrained_welfare":
                if slack_per_episode:
                    # Per-inner-episode slack, shape [M]. `runner_welfare_evo`
                    # evaluates the slack per population member, so its
                    # quadratic penalty varies across the fitness batch. A
                    # single meta-trajectory scalar (the previous behaviour)
                    # makes the penalty a pure gradient-magnitude multiplier
                    # with no steering signal; resolving it per episode gives
                    # the penalty back its discrimination along the M axis,
                    # which is the axis the shaper actually controls.
                    slack_1 = ep_rewards_1 - _objective_state[6]
                    slack_2 = ep_rewards_2 - _objective_state[7]
                else:
                    slack_1 = ep_rewards_1.mean() - _objective_state[6]
                    slack_2 = ep_rewards_2.mean() - _objective_state[7]

                # dL/dR_k = 1 + mu_k + rho_k * max(0, v_ref_k - R_k).
                w1 = 1.0 + _objective_state[2] + _objective_state[4] * jnp.maximum(
                    0.0, -slack_1
                )
                w2 = 1.0 + _objective_state[3] + _objective_state[5] * jnp.maximum(
                    0.0, -slack_2
                )
                if not weight_normalization:
                    # `weight_max` exists only to bound the reward scale fed to
                    # the critic. Normalization already does that exactly --
                    # after dividing by the mean weight, w1 + w2 == 2
                    # identically -- so capping BEFORE normalizing can only
                    # distort the ratio, which is the whole informative content
                    # of the Lagrangian direction. Worse, once both weights hit
                    # the same cap the ratio collapses to 1:1 and every
                    # per-episode and per-player distinction is erased.
                    # Measured on IPD at v_ref=(-15,-15): w_shaper pinned at
                    # exactly 10.0 (= weight_max) in every episode while
                    # w_co-player sat at 8.6, and that run drifted into the
                    # all-defect basin. So the cap applies only on the
                    # un-normalized path, where it is still the sole protection
                    # against an unbounded reward scale.
                    weight_cap = _objective_state[8]
                    w1 = jnp.minimum(w1, weight_cap)
                    w2 = jnp.minimum(w2, weight_cap)
            else:
                w1 = _objective_state[0]
                w2 = _objective_state[1]

            # Broadcast over traj rewards [M, T, num_opps, num_envs]: a
            # per-episode weight is [M] -> [M, 1, 1, 1]; a scalar stays scalar.
            def _bcast(w):
                w = jnp.asarray(w)
                return w.reshape(w.shape + (1,) * (traj_1.rewards.ndim - w.ndim))

            objective_rewards = (
                _bcast(w1) * traj_1.rewards + _bcast(w2) * traj_2.rewards
            )
            if coala_objective == "constrained_welfare" and weight_normalization:
                # Keep the reward scale the critic regresses STATIONARY across
                # iterations. Without this, every change in mu/rho rescales the
                # whole reward stream, and `reward_rescaling` (a constant tuned
                # for w ~ 1) cannot track it; the critic is then always chasing
                # a target of the wrong magnitude and `clip_value` throttles how
                # fast it can catch up. Dividing by the mean weight preserves
                # the *relative* weighting between the two players — which is
                # the entire content of the Lagrangian gradient direction — and
                # discards only the common magnitude, which Adam would rescale
                # away anyway if the critic were exact. At w=(1,1) the divisor
                # is 1.0, so the plain-welfare case is unchanged.
                objective_rewards = objective_rewards / _bcast(
                    0.5 * (w1 + w2)
                )

            # Mean over M for logging, so the reported weights have the same
            # shape in both the per-episode and scalar-slack settings.
            objective_weights = jnp.asarray(
                [jnp.mean(w1), jnp.mean(w2)]
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
            # Unused by the constrained branch of `_rollout`, which rebuilds
            # the weights from mu/rho/v_ref against the realised slack; kept so
            # indices 0/1 mean "the weights" in every mode.
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
                self.weight_max if self.weight_max > 0.0 else jnp.inf,
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
        if self.args.get("coala_objective", "selfish") == "constrained_welfare":
            print("Constrained welfare (augmented Lagrangian):")
            print(
                f"  v_ref: shaper {self.v_ref_shaper:.4f} | "
                f"co-player {self.v_ref_opponent:.4f}"
            )
            print(f"  dual_lr: {self.dual_lr} | mu_max: {self.mu_max}")
            print(
                f"  rho: {self.rho1} -> x{self.rho_multiplier} every "
                f"{self.rho_patience} violations, capped at {self.rho_max}"
            )
            print(
                f"  weight_max: {self.weight_max} | "
                f"weight_normalization: {self.weight_normalization} | "
                f"slack_per_episode: {self.slack_per_episode}"
            )

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
                # Dual ascent, identical to `runner_welfare_evo`, except that
                # mu is clamped above. An infeasible v_ref makes the dual
                # variable a pure integrator with no primal response, so it
                # winds up without bound: violation -> mu up -> reward scale up
                # -> critic de-calibrated -> policy randomises -> larger
                # violation. `mu_max` breaks that feedback loop. If mu sits at
                # mu_max for long stretches, the constraint is infeasible and
                # v_ref should be relaxed rather than the cap raised.
                self.mu1 = min(
                    self.mu_max, max(0.0, self.mu1 - self.dual_lr * slack_1)
                )
                self.mu2 = min(
                    self.mu_max, max(0.0, self.mu2 - self.dual_lr * slack_2)
                )

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
                    # mu/rho above are POST-update (they take effect next
                    # iteration); these are the weights that produced the
                    # returns printed above. Printing only the former made the
                    # two look inconsistent.
                    print(
                        f"  weights     : w_shaper "
                        f"{float(objective_weights[0]):.4f} | "
                        f"w_co-player {float(objective_weights[1]):.4f}"
                        + ("  (mean over M)" if self.slack_per_episode else "")
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
                            # Watch these two: mu pinned at mu_max, or a weight
                            # pinned at weight_max, means the constraint is
                            # infeasible at this v_ref — relax v_ref rather
                            # than raising the cap.
                            "train/lagrangian/mu_max": self.mu_max,
                            "train/lagrangian/weight_max": self.weight_max,
                            "train/lagrangian/mu_at_cap_shaper": float(
                                self.mu1 >= self.mu_max
                            ),
                            "train/lagrangian/mu_at_cap_opponent": float(
                                self.mu2 >= self.mu_max
                            ),
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
