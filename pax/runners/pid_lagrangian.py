"""PID Lagrangian multiplier controller.

Stooke, Achiam & Abbeel, "Responsive Safety in Reinforcement Learning by PID
Lagrangian Methods", ICML 2020. Reference implementation: OmniSafe's
``PIDLagrangian``.

Plain dual ascent (RCPO, Tessler et al. 2019) is a pure integrator on the
constraint violation: lam only moves once a violation has already accumulated,
so it lags into the violation and then overshoots coming out. Adding a
proportional term (reacts to the current violation) and a derivative term
(reacts to the violation getting *worse*) makes lam respond before the primal
has been driven deep into the infeasible region.

Sign convention here. The constraint is a FLOOR on a return:

    R_s >= tau

Written as a cost constraint ``J_c <= d`` with ``J_c = -R_s`` and ``d = -tau``,
the violation is

    delta = J_c - d = tau - R_s          (positive == violated)

so a satisfied constraint gives delta < 0 and drains the integrator. This
module is deliberately free of JAX: lam is a scalar updated once per
meta-trajectory on the host, and keeping it in plain Python makes it trivial to
test, log, and checkpoint.

Setting ``kp = kd = 0.0`` reduces the update to ``lam <- max(0, lam + ki*delta)``,
which is exactly RCPO dual ascent with step size ``ki``. That is intentional:
the same code path provides the dual-ascent baseline for ablations.
"""

from typing import Dict, Optional


class PIDLagrangian:
    """Scalar multiplier for one inequality constraint ``R >= tau``."""

    def __init__(
        self,
        tau: float,
        kp: float = 0.0,
        ki: float = 0.01,
        kd: float = 0.0,
        lam_init: float = 0.5,
        lam_max: Optional[float] = None,
        ema_beta: float = 0.0,
    ):
        """
        Args:
          tau: the floor the constrained return must clear.
          kp, ki, kd: proportional / integral / derivative gains. kp = kd = 0
            recovers RCPO dual ascent with learning rate ki.
          lam_init: initial multiplier. Deliberately > 0 by default -- early in
            meta-training the constraint is often satisfied by accident, and a
            multiplier initialised at 0 decays further, so by the time real
            violations appear the shaper has already learnt a self-sacrificing
            policy and lam has to climb from nothing to undo it.
          lam_max: optional ceiling. None leaves lam unbounded, which is the
            textbook method; a finite value trades exact feasibility for a
            bound on how far the objective can be tilted.
          ema_beta: if > 0, the controller sees an exponential moving average
            of the measured return instead of the raw per-iteration value.
            COALA-PG meta-trajectory returns are noisy and a derivative term on
            raw noise is mostly noise, so smoothing matters more here than in
            single-agent safe RL.
        """
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        self.tau = float(tau)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.lam_max = None if lam_max is None else float(lam_max)
        self.ema_beta = float(ema_beta)

        # The integrator holds the multiplier that pure dual ascent would
        # produce; lam is that plus the P and D corrections.
        self._integral = float(lam_init)
        self.lam = float(lam_init)
        self._prev_return: Optional[float] = None
        self._smoothed_return: Optional[float] = None

        # Diagnostics for the last update.
        self.delta = 0.0
        self.p_term = 0.0
        self.d_term = 0.0

    def update(self, measured_return: float) -> float:
        """Consume one measurement of the constrained return; return new lam."""
        r = float(measured_return)

        if self.ema_beta > 0.0:
            if self._smoothed_return is None:
                self._smoothed_return = r
            else:
                self._smoothed_return = (
                    self.ema_beta * self._smoothed_return
                    + (1.0 - self.ema_beta) * r
                )
            r_eff = self._smoothed_return
        else:
            r_eff = r

        # Violation: positive when the return is below the floor.
        delta = self.tau - r_eff

        # Integral: the dual-ascent term. Clamped at zero so a long stretch of
        # satisfied constraint cannot bank negative multiplier "credit" that
        # would delay the response to the next violation.
        self._integral = max(0.0, self._integral + self.ki * delta)

        # Derivative on the COST, i.e. positive when the return is falling.
        # One-sided: we want lam to anticipate a developing violation, but not
        # to be dragged down merely because things are improving -- the
        # integral already handles the release.
        if self._prev_return is None:
            d_raw = 0.0
        else:
            d_raw = self._prev_return - r_eff
        self._prev_return = r_eff

        self.delta = delta
        self.p_term = self.kp * delta
        self.d_term = self.kd * max(0.0, d_raw)

        lam = self._integral + self.p_term + self.d_term
        lam = max(0.0, lam)
        if self.lam_max is not None:
            lam = min(lam, self.lam_max)
        self.lam = lam
        return self.lam

    def state_dict(self) -> Dict[str, float]:
        """Everything needed to resume the controller mid-run."""
        return {
            "lam": self.lam,
            "integral": self._integral,
            "prev_return": (
                float("nan")
                if self._prev_return is None
                else self._prev_return
            ),
            "smoothed_return": (
                float("nan")
                if self._smoothed_return is None
                else self._smoothed_return
            ),
        }

    def metrics(self) -> Dict[str, float]:
        return {
            "lagrangian/lam": self.lam,
            "lagrangian/violation": self.delta,
            "lagrangian/integral": self._integral,
            "lagrangian/p_term": self.p_term,
            "lagrangian/d_term": self.d_term,
            "lagrangian/tau": self.tau,
        }
