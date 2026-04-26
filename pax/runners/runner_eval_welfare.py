import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

import wandb
from pax.utils import load
from pax.watchers import cg_visitation, ipd_visitation

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
    """
    Welfare evaluation runner. Mirrors EvalHardstopRunner's structure but
    drops the two-phase hardstop logic — the opponent always learns. Adds
    tracking of joint welfare, individual returns, and IR (individual
    rationality) constraint slack against pre-calibrated reference values
    (welfare.v_ref_shaper / welfare.v_ref_opponent).

    Eval protocol (one trial per job):
      - Initialise a fresh opponent.
      - Opponent learns over `num_outer_steps` inner episodes against the
        frozen shaper.
      - Per-outer-step rewards are logged so the converged reward (last few
        episodes) can be inspected in wandb.

    Multiple seeds (separate jobs) provide the statistical sample of fresh
    NLs.

    Args:
        agents (Tuple[agents]):
            The set of agents that will run in the experiment. Note, ordering
            is important for logic used in the class.
        env (gymnax.envs.Environment):
            The environment that the agents will run in.
        args (NamedTuple):
            A tuple of experiment arguments used (usually provided by HydraConfig).
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

        # Welfare / IR constraint reference values
        self.v_ref_shaper = args.welfare.v_ref_shaper
        self.v_ref_opponent = args.welfare.v_ref_opponent

        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )

        # VMAP for num opps: we vmap over the rng but not params
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(
                env.step, (0, 0, 0, None), 0  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))

        agent1, agent2 = agents

        if args.agent1 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent1.batch_init = jax.jit(jax.vmap(agent1.make_initial_state))
        else:
            # batch MemoryState not TrainingState
            agent1.batch_init = jax.vmap(
                agent1.make_initial_state,
                (None, 0),
                (None, 0),
            )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )

        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # batch all for Agent2
        if args.agent2 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent2.batch_init = jax.jit(jax.vmap(agent2.make_initial_state))
        else:
            agent2.batch_init = jax.vmap(
                agent2.make_initial_state, (0, None), 0
            )
        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy))
        agent2.batch_reset = jax.jit(
            jax.vmap(agent2.reset_memory, (0, None), 0), static_argnums=1
        )
        agent2.batch_update = jax.jit(jax.vmap(agent2.update, (1, 0, 0, 0), 0))

        if args.agent1 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
            agent1._state, agent1._mem = agent1.batch_init(
                agent1._state.random_key, init_hidden
            )

        if args.agent2 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))
            agent2._state, agent2._mem = agent2.batch_init(
                jax.random.split(agent2._state.random_key, args.num_opps),
                init_hidden,
            )

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
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

            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            # a1_rng = rngs[:, :, 1, :]
            # a2_rng = rngs[:, :, 2, :]
            rngs = rngs[:, :, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                obs1,
                a1_mem,
            )
            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state,
                obs2,
                a2_mem,
            )
            (next_obs1, next_obs2), env_state, rewards, done, info = env.step(
                env_rng,
                env_state,
                (a1, a2),
                env_params,
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
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            """Runner for trial — opponent learns (update applied)."""
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=self.args.num_inner_steps,
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
            # MFOS-style shapers take a meta-action between episodes
            if args.agent1 in ["MFOS", "WelfareShaper"]:
                a1_mem = agent1.meta_policy(a1_mem)

            # update second agent
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1],
                obs2,
                a2_state,
                a2_mem,
            )
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

        self.rollout = jax.jit(_outer_rollout)

    def run_loop(self, env, env_params, agents, num_episodes, watchers):
        """Run welfare evaluation of agents in environment"""
        print("Welfare Evaluation")
        print("-----------------------")
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        # Only call wandb.restore for online mode AND a wandb-relative
        # model path (absolute local paths cannot be restored).
        if (
            watchers
            and self.args.wandb.get("mode", "online")
            not in ["offline", "disabled"]
            and not os.path.isabs(self.model_path)
        ):
            wandb.restore(
                name=self.model_path,
                run_path=self.run_path,
                root=os.getcwd(),
            )
        pretrained_params = load(self.model_path)
        a1_state = a1_state._replace(params=pretrained_params)
        print(f"Loaded pretrained shaper from: {self.model_path}")
        print(f"v_ref_shaper:   {self.v_ref_shaper:.4f}")
        print(f"v_ref_opponent: {self.v_ref_opponent:.4f}")

        num_iters = max(
            int(num_episodes / (self.args.num_envs * self.args.num_opps)), 1
        )
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Log Interval {log_interval}")
        print(
            f"Num outer episodes per trial: {self.args.num_outer_steps} "
            f"(opponent learns every episode)"
        )

        # RNG are the same for num_opps but different for num_envs
        rngs = jnp.concatenate(
            [jax.random.split(rng, self.args.num_envs)] * self.args.num_opps
        ).reshape((self.args.num_opps, self.args.num_envs, -1))

        # run actual loop (one iteration per trial; num_iters typically 1)
        print("num episodes", num_episodes)
        for trial_idx in range(num_episodes):

            obs, env_state = env.reset(rngs, env_params)
            rewards = [
                jnp.zeros((self.args.num_opps, self.args.num_envs)),
                jnp.zeros((self.args.num_opps, self.args.num_envs)),
            ]

            if self.args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(obs[1])
            elif self.args.env_type in ["meta"]:
                # meta-experiments - init 2nd agent per trial
                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(rng, self.num_opps), a2_mem.hidden
                )

            # Single phase: opponent learns for `num_outer_steps` outer episodes
            vals, stack = jax.lax.scan(
                self.rollout,
                (
                    rngs,
                    *obs,
                    *rewards,
                    a1_state,
                    a1_mem,
                    a2_state,
                    a2_mem,
                    env_state,
                    env_params,
                ),
                None,
                length=self.args.num_outer_steps,
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
            traj_1, traj_2, a2_metrics = stack

            # reset second agent memory
            a2_mem = agent2.batch_reset(a2_mem, False)

            # ---- Per-inner-episode (per-outer-step) episodic rewards ----
            # Shape: [num_outer_steps, num_inner_steps, num_opps, num_envs]
            # → sum over inner_steps, mean over envs → [num_outer_steps, num_opps]
            traj_1_per_step = traj_1.rewards.sum(axis=1).mean(axis=2)
            traj_2_per_step = traj_2.rewards.sum(axis=1).mean(axis=2)
            for step_idx in range(traj_1_per_step.shape[0]):
                r1_step = float(traj_1_per_step[step_idx].mean())
                r2_step = float(traj_2_per_step[step_idx].mean())
                wandb.log(
                    {
                        "outer_step": step_idx,
                        "eval/per_step/reward/player_1": r1_step,
                        "eval/per_step/reward/player_2": r2_step,
                        "eval/per_step/welfare": r1_step + r2_step,
                        "eval/per_step/slack_shaper": r1_step
                        - self.v_ref_shaper,
                        "eval/per_step/slack_opponent": r2_step
                        - self.v_ref_opponent,
                    }
                )

            # ---- Trial mean (overall, episodic) ----
            mean_r1 = float(traj_1.rewards.sum(axis=1).mean())
            mean_r2 = float(traj_2.rewards.sum(axis=1).mean())
            welfare = mean_r1 + mean_r2
            slack_shaper = mean_r1 - self.v_ref_shaper
            slack_opponent = mean_r2 - self.v_ref_opponent

            self.train_episodes += 1
            if trial_idx % log_interval == 0:
                print(f"Trial {trial_idx}/{num_episodes}")
                if self.args.env_id == "coin_game":
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(),
                        self.cg_stats(env_state),
                    )

                elif self.args.env_type in [
                    "meta",
                    "sequential",
                ]:
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(),
                        self.ipd_stats(
                            traj_1.observations,
                            traj_1.actions,
                            obs1,
                        ),
                    )

                else:
                    env_stats = {}

                print(f"Env Stats: {env_stats}")
                print(
                    f"  Trial mean: R1={mean_r1:.4f}  R2={mean_r2:.4f}  "
                    f"Welfare={welfare:.4f}"
                )
                print(
                    f"  Constraint slack (trial mean): "
                    f"shaper={slack_shaper:.4f}  opponent={slack_opponent:.4f}"
                )
                print()

                if watchers:
                    # metrics [outer_timesteps, num_opps]
                    flattened_metrics = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                    )
                    agent2._logger.metrics = (
                        agent2._logger.metrics | flattened_metrics
                    )

                    for watcher, agent in zip(watchers, agents):
                        watcher(agent)
                    wandb.log(
                        {
                            "trials": self.train_episodes,
                            "eval/trial_mean/reward/player_1": mean_r1,
                            "eval/trial_mean/reward/player_2": mean_r2,
                            "eval/trial_mean/welfare": welfare,
                            "eval/trial_mean/slack_shaper": slack_shaper,
                            "eval/trial_mean/slack_opponent": slack_opponent,
                        }
                        | env_stats,
                    )

        agents[0]._state = a1_state
        agents[1]._state = a2_state
        return agents
