"""Welfare Shaper Evaluation Runner.

Freezes a pretrained welfare shaper (agent1) and lets a fresh naive learner
(agent2) learn while interacting with it over many episodes.  Tracks joint
welfare, individual returns, and IR constraint satisfaction.

Supports both WelfareShaper (MFOS-based, needs meta_policy between episodes)
and WelfareShaperAtt (attention-based, no meta_policy).
"""

import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

import wandb
from pax.utils import load
from pax.watchers import cg_visitation, ipd_visitation, ipditm_stats

MAX_WANDB_CALLS = 10000


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class WelfareEvalRunner:
    """Evaluation runner for welfare shapers.

    Loads pretrained welfare shaper params (frozen), initialises a fresh
    opponent, then runs multiple episodes where only the opponent learns.
    Logs per-episode rewards, joint welfare, and IR constraint satisfaction.

    Args:
        agents: (shaper, opponent) pair.
        env: Environment.
        args: Hydra experiment config.
    """

    def __init__(self, agents, env, args):
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_opps = args.num_opps
        self.random_key = jax.random.PRNGKey(args.seed)
        self.run_path = args.run_path
        self.model_path = args.model_path
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)

        # v_ref for IR constraint tracking (set from config or default 0)
        welfare_cfg = getattr(args, "welfare", None)
        self.v_ref_shaper = (
            welfare_cfg.v_ref_shaper if welfare_cfg and hasattr(welfare_cfg, "v_ref_shaper") else 0.0
        )
        self.v_ref_opponent = (
            welfare_cfg.v_ref_opponent if welfare_cfg and hasattr(welfare_cfg, "v_ref_opponent") else 0.0
        )

        # VMAP for num envs
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(env.step, (0, 0, 0, None), 0)
        # VMAP for num opps
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(jax.vmap(env.step, (0, 0, 0, None), 0))

        self.split = jax.vmap(
            jax.vmap(jax.random.split, (0, None)), (0, None)
        )

        agent1, agent2 = agents

        # ---- Batch agent1 (shaper, frozen) ----
        if args.agent1 == "NaiveEx":
            agent1.batch_init = jax.jit(jax.vmap(agent1.make_initial_state))
        else:
            agent1.batch_init = jax.vmap(
                agent1.make_initial_state, (None, 0), (None, 0)
            )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0),
            static_argnums=1,
        )
        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # ---- Batch agent2 (opponent, learns) ----
        if args.agent2 == "NaiveEx":
            agent2.batch_init = jax.jit(jax.vmap(agent2.make_initial_state))
        else:
            agent2.batch_init = jax.vmap(
                agent2.make_initial_state, (0, None), 0
            )
        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy))
        agent2.batch_reset = jax.jit(
            jax.vmap(agent2.reset_memory, (0, None), 0),
            static_argnums=1,
        )
        agent2.batch_update = jax.jit(
            jax.vmap(agent2.update, (1, 0, 0, 0), 0)
        )

        # ---- Init agent hidden states ----
        if args.agent1 != "NaiveEx":
            init_hidden = jnp.tile(
                agent1._mem.hidden, (args.num_opps, 1, 1)
            )
            agent1._state, agent1._mem = agent1.batch_init(
                agent1._state.random_key, init_hidden
            )

        if args.agent2 != "NaiveEx":
            init_hidden = jnp.tile(
                agent2._mem.hidden, (args.num_opps, 1, 1)
            )
            agent2._state, agent2._mem = agent2.batch_init(
                jax.random.split(agent2._state.random_key, args.num_opps),
                init_hidden,
            )

        # ---- Inner rollout (one timestep) ----
        def _inner_rollout(carry, unused):
            (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
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

            (next_obs1, next_obs2), env_state, rewards, done, info = env.step(
                env_rng, env_state, (a1, a2), env_params
            )

            traj1 = Sample(
                obs1, a1, rewards[0],
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done, a1_mem.hidden,
            )
            traj2 = Sample(
                obs2, a2, rewards[1],
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done, a2_mem.hidden,
            )

            return (
                rngs, next_obs1, next_obs2, rewards[0], rewards[1],
                a1_state, new_a1_mem, a2_state, new_a2_mem,
                env_state, env_params,
            ), (traj1, traj2)

        # ---- Outer rollout (one episode) ----
        def _outer_rollout(carry, unused):
            vals, trajectories = jax.lax.scan(
                _inner_rollout, carry, None,
                length=self.args.num_inner_steps,
            )
            (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ) = vals

            # Meta-action for MFOS-based welfare shaper
            if args.agent1 in ["WelfareShaper", "MFOS"]:
                a1_mem = agent1.meta_policy(a1_mem)

            # Update opponent (agent2 learns)
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1], obs2, a2_state, a2_mem,
            )

            return (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ), (*trajectories, a2_metrics)

        self.rollout = jax.jit(_outer_rollout)

    def run_loop(self, env, env_params, agents, num_episodes, watchers):
        """Run evaluation: frozen shaper vs learning opponent."""
        print("Welfare Evaluation")
        print("-----------------------")
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        # Load pretrained shaper params and freeze
        if self.args.wandb.get("mode", "online") not in ["offline", "disabled"]:
            if watchers:
                wandb.restore(
                    name=self.model_path,
                    run_path=self.run_path,
                    root=os.getcwd(),
                )
        pretrained_params = load(self.model_path)
        a1_state = a1_state._replace(params=pretrained_params)
        print(f"Loaded pretrained shaper from: {self.model_path}")
        print(f"v_ref_shaper: {self.v_ref_shaper:.4f}")
        print(f"v_ref_opponent: {self.v_ref_opponent:.4f}")

        num_iters = max(
            int(num_episodes / (self.args.num_envs * self.args.num_opps)), 1
        )
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Log Interval {log_interval}")
        print(f"Num outer steps per trial: {self.args.num_outer_steps}")

        # Accumulators for summary statistics
        all_r1 = []
        all_r2 = []
        all_welfare = []

        for i in range(num_episodes):
            rng, rng_reset = jax.random.split(rng)
            rngs = jnp.concatenate(
                [jax.random.split(rng_reset, self.args.num_envs)]
                * self.args.num_opps
            ).reshape((self.args.num_opps, self.args.num_envs, -1))

            obs, env_state = env.reset(rngs, env_params)
            rewards = [
                jnp.zeros((self.args.num_opps, self.args.num_envs)),
                jnp.zeros((self.args.num_opps, self.args.num_envs)),
            ]

            # Re-init opponent each episode (fresh naive learner)
            agent2_reset_interval = getattr(
                self.args, "agent2_reset_interval", 1
            )
            if i % agent2_reset_interval == 0:
                if self.args.agent2 == "NaiveEx":
                    a2_state, a2_mem = agent2.batch_init(obs[1])
                elif self.args.env_type in ["meta"]:
                    a2_state, a2_mem = agent2.batch_init(
                        jax.random.split(rng_reset, self.num_opps),
                        a2_mem.hidden,
                    )

            # Reset shaper memory at start of each trial
            a1_mem = agent1.batch_reset(a1_mem, False)

            # Run trial: scan over outer episodes
            vals, stack = jax.lax.scan(
                self.rollout,
                (
                    rngs, *obs, *rewards,
                    a1_state, a1_mem, a2_state, a2_mem,
                    env_state, env_params,
                ),
                None,
                length=self.args.num_outer_steps,
            )

            (
                rngs, obs1, obs2, r1, r2,
                a1_state,
                _a1_mem,  # don't carry shaper memory across trials
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = vals
            traj_1, traj_2, a2_metrics = stack

            # Reset opponent memory for next trial
            a2_mem = agent2.batch_reset(a2_mem, False)

            # ---- Compute episode statistics ----
            rewards_1 = traj_1.rewards.mean()
            rewards_2 = traj_2.rewards.mean()
            welfare = float(rewards_1) + float(rewards_2)

            all_r1.append(float(rewards_1))
            all_r2.append(float(rewards_2))
            all_welfare.append(welfare)

            # ---- Env-specific stats ----
            if self.args.env_id == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.item(), self.cg_stats(env_state)
                )
            elif self.args.env_id in [
                "iterated_matrix_game",
            ]:
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.item(),
                    self.ipd_stats(
                        traj_1.observations, traj_1.actions, obs1
                    ),
                )
            else:
                env_stats = {}

            # ---- Constraint slack ----
            slack_shaper = float(rewards_1) - self.v_ref_shaper
            slack_opponent = float(rewards_2) - self.v_ref_opponent
            ir_satisfied = slack_shaper >= 0 and slack_opponent >= 0

            # ---- Logging ----
            self.train_episodes += 1
            if i % log_interval == 0:
                print(f"Episode {i}/{num_episodes}")
                print(
                    f"  R1 (shaper): {float(rewards_1):.4f} | "
                    f"R2 (opponent): {float(rewards_2):.4f} | "
                    f"Welfare: {welfare:.4f}"
                )
                print(
                    f"  Constraint slack: shaper={slack_shaper:.4f}  "
                    f"opponent={slack_opponent:.4f}  "
                    f"IR satisfied: {ir_satisfied}"
                )
                if env_stats:
                    printable = {
                        k: v for k, v in env_stats.items()
                        if not k.startswith("states")
                    }
                    print(f"  Env Stats: {printable}")
                print()

            if watchers:
                # Flatten a2 metrics for logging
                flattened_metrics = jax.tree_util.tree_map(
                    lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                )
                agent2._logger.metrics = (
                    agent2._logger.metrics | flattened_metrics
                )

                for watcher, agent in zip(watchers, agents):
                    watcher(agent)

                wandb_log = {
                    "episodes": self.train_episodes,
                    "eval/reward/player_1": float(rewards_1),
                    "eval/reward/player_2": float(rewards_2),
                    "eval/welfare": welfare,
                    "eval/constraint_slack/shaper": slack_shaper,
                    "eval/constraint_slack/opponent": slack_opponent,
                    "eval/ir_satisfied": int(ir_satisfied),
                }
                wandb_log.update(env_stats)
                wandb.log(wandb_log)

        # ---- Summary ----
        mean_r1 = sum(all_r1) / len(all_r1) if all_r1 else 0.0
        mean_r2 = sum(all_r2) / len(all_r2) if all_r2 else 0.0
        mean_welfare = sum(all_welfare) / len(all_welfare) if all_welfare else 0.0
        print("=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Mean R1 (shaper):   {mean_r1:.4f}")
        print(f"Mean R2 (opponent): {mean_r2:.4f}")
        print(f"Mean Welfare:       {mean_welfare:.4f}")
        print(f"v_ref_shaper:       {self.v_ref_shaper:.4f}")
        print(f"v_ref_opponent:     {self.v_ref_opponent:.4f}")
        print(
            f"IR shaper:          {'PASS' if mean_r1 >= self.v_ref_shaper else 'FAIL'} "
            f"(slack={mean_r1 - self.v_ref_shaper:.4f})"
        )
        print(
            f"IR opponent:        {'PASS' if mean_r2 >= self.v_ref_opponent else 'FAIL'} "
            f"(slack={mean_r2 - self.v_ref_opponent:.4f})"
        )
        print("=" * 60)

        if watchers:
            wandb.log({
                "eval/summary/mean_r1": mean_r1,
                "eval/summary/mean_r2": mean_r2,
                "eval/summary/mean_welfare": mean_welfare,
                "eval/summary/ir_slack_shaper": mean_r1 - self.v_ref_shaper,
                "eval/summary/ir_slack_opponent": mean_r2 - self.v_ref_opponent,
            })

        agents[0]._state = a1_state
        agents[1]._state = a2_state
        return agents
