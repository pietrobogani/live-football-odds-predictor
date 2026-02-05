"""
Poisson/Skellam model for predicting how football match probabilities
change in real-time when a goal is scored.

Given live 1X2 and Over/Under market prices, calibrates Poisson rates via:
  O/U prices -> m (total remaining goals)
  1X2 prices -> q (home's share of remaining goals)
  lambda_home = m * q, lambda_away = m * (1 - q)

Then uses the Skellam distribution (difference of two Poissons) to compute
new Home/Draw/Away probabilities after a hypothetical goal.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from scipy.special import iv as bessel_i
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize


@dataclass
class Probabilities:
    """1X2 probabilities."""
    home_win: float
    draw: float
    away_win: float

    def normalize(self) -> 'Probabilities':
        total = self.home_win + self.draw + self.away_win
        if total == 0:
            return Probabilities(1/3, 1/3, 1/3)
        return Probabilities(self.home_win / total, self.draw / total, self.away_win / total)


# -- Skellam distribution -----------------------------------------------------

def skellam_pmf(k: int, lambda1: float, lambda2: float) -> float:
    """
    P(X - Y = k) where X ~ Poisson(lambda1), Y ~ Poisson(lambda2).

    Uses modified Bessel function of the first kind.
    """
    if lambda1 <= 0 and lambda2 <= 0:
        return 1.0 if k == 0 else 0.0
    if lambda1 <= 0:
        return poisson_dist.pmf(-k, lambda2) if k <= 0 else 0.0
    if lambda2 <= 0:
        return poisson_dist.pmf(k, lambda1) if k >= 0 else 0.0

    sqrt_prod = math.sqrt(lambda1 * lambda2)
    log_prob = -(lambda1 + lambda2) + (k / 2) * math.log(lambda1 / lambda2)
    return math.exp(log_prob) * bessel_i(abs(k), 2 * sqrt_prod)


# -- Probability calculations -------------------------------------------------

def compute_1x2_from_poisson(lambda_home: float, lambda_away: float,
                              current_diff: int, max_goals: int = 15) -> Probabilities:
    """Compute 1X2 probabilities from Poisson params and current score difference."""
    p_home, p_draw, p_away = 0.0, 0.0, 0.0
    for d in range(-max_goals, max_goals + 1):
        prob = skellam_pmf(d, lambda_home, lambda_away)
        final = current_diff + d
        if final > 0: p_home += prob
        elif final == 0: p_draw += prob
        else: p_away += prob
    return Probabilities(p_home, p_draw, p_away)


def compute_over_under(lambda_home: float, lambda_away: float,
                       current_total: int, threshold: float) -> Tuple[float, float]:
    """Compute (P(over), P(under)) for a goal line given current total and Poisson params."""
    goals_needed = max(0, math.ceil(threshold) - current_total)
    if goals_needed <= 0:
        return (1.0, 0.0)
    lam = lambda_home + lambda_away
    if lam <= 0.001:
        return (0.0, 1.0)
    p_under = poisson_dist.cdf(goals_needed - 1, lam)
    return (1 - p_under, p_under)


# -- Dual calibration (O/U + 1X2) ---------------------------------------------

def solve_m_from_over_probability(target_over: float, goals_needed: int) -> float:
    """Binary search for m (total remaining goals) given P(Over) and goals needed."""
    if target_over <= 0.001: return 0.01
    if target_over >= 0.999: return 10.0
    low, high = 0.01, 10.0
    for _ in range(100):
        mid = (low + high) / 2
        p_over = 1 - poisson_dist.cdf(goals_needed - 1, mid)
        if abs(p_over - target_over) < 0.0001:
            return mid
        if p_over < target_over: low = mid
        else: high = mid
    return (low + high) / 2


def calibrate_m_from_ou(ou_prices: Dict[float, float], current_goals: int) -> Optional[float]:
    """
    Calibrate m (total remaining goals) from O/U market prices.

    Weighted average of implied m values from each O/U line.
    Skips settled markets (goals_needed <= 0) and extreme prices (<15% or >85%).
    Weights: prices closer to 50% are more informative.
    """
    lines = []
    for line, price in ou_prices.items():
        goals_needed = math.ceil(line) - current_goals
        if goals_needed <= 0 or price < 0.15 or price > 0.85:
            continue
        try:
            m = solve_m_from_over_probability(price, goals_needed)
            weight = max(0.1, 1 - 2 * abs(price - 0.50))
            lines.append((m, weight))
        except Exception:
            continue
    if not lines:
        return None
    total_w = sum(w for _, w in lines)
    return sum(m * w for m, w in lines) / total_w


def calibrate_q_from_1x2(target: Probabilities, m: float, current_diff: int) -> Optional[float]:
    """
    Find q (home's share of remaining goals) such that Skellam(m*q, m*(1-q))
    best matches the target 1X2 probabilities.
    """
    if m <= 0.001:
        return 0.5

    def error(q_arr):
        q = q_arr[0]
        if q < 0.05 or q > 0.95:
            return 1e10
        p = compute_1x2_from_poisson(m * q, m * (1 - q), current_diff)
        return (p.home_win - target.home_win)**2 + (p.draw - target.draw)**2 + (p.away_win - target.away_win)**2

    initial = 0.55 if target.home_win > target.away_win else 0.45
    try:
        result = minimize(error, x0=[initial], method='Nelder-Mead', options={'maxiter': 500, 'xatol': 0.001})
        return max(0.05, min(0.95, result.x[0]))
    except Exception:
        return None


def calibrate_poisson(target_1x2: Probabilities, ou_prices: Dict[float, float],
                      current_goals: int, current_diff: int) -> Optional[Tuple[float, float]]:
    """
    Dual calibration: O/U -> m, then 1X2 -> q, then lambda = m*q.
    Returns (lambda_home, lambda_away) or None if calibration fails.
    """
    m = calibrate_m_from_ou(ou_prices, current_goals)
    if m is None or not (0.05 <= m <= 8.0):
        return None
    q = calibrate_q_from_1x2(target_1x2, m, current_diff)
    if q is None or not (0.05 <= q <= 0.95):
        return None
    return (m * q, m * (1 - q))


# -- Prediction engine ---------------------------------------------------------

@dataclass
class GoalImpact:
    """Predicted probability changes after a goal."""
    new_1x2: Probabilities
    delta_home_win: float
    delta_draw: float
    delta_away_win: float
    new_over_2_5: float
    delta_over_2_5: float


class FootballPredictor:
    """
    Predicts how match probabilities change when a goal is scored.

    Usage:
        predictor = FootballPredictor()
        predictor.update(minute=30, home_goals=0, away_goals=0,
                         market_1x2=Probabilities(0.50, 0.28, 0.22),
                         ou_prices={2.5: 0.55, 3.5: 0.30})
        impact = predictor.predict_goal_impact(home_scores=True)
    """

    def __init__(self):
        self.home_goals = 0
        self.away_goals = 0
        self.current_1x2: Optional[Probabilities] = None
        self.lambda_home: Optional[float] = None
        self.lambda_away: Optional[float] = None

    def update(self, minute: int, home_goals: int, away_goals: int,
               market_1x2: Probabilities, ou_prices: Dict[float, float]) -> None:
        """Update model with current match state and market prices."""
        self.home_goals = home_goals
        self.away_goals = away_goals
        self.current_1x2 = market_1x2.normalize()
        params = calibrate_poisson(
            self.current_1x2, ou_prices,
            current_goals=home_goals + away_goals,
            current_diff=home_goals - away_goals,
        )
        if params:
            self.lambda_home, self.lambda_away = params
        else:
            self.lambda_home = self.lambda_away = None

    def predict_goal_impact(self, home_scores: bool) -> GoalImpact:
        """Predict new probabilities if home (True) or away (False) scores next."""
        if self.lambda_home is None or self.current_1x2 is None:
            raise ValueError("Must call update() first")

        new_diff = (self.home_goals - self.away_goals) + (1 if home_scores else -1)
        new_total = self.home_goals + self.away_goals + 1

        new_1x2 = compute_1x2_from_poisson(self.lambda_home, self.lambda_away, new_diff)
        cur_o25, _ = compute_over_under(self.lambda_home, self.lambda_away, self.home_goals + self.away_goals, 2.5)
        new_o25, _ = compute_over_under(self.lambda_home, self.lambda_away, new_total, 2.5)

        return GoalImpact(
            new_1x2=new_1x2,
            delta_home_win=new_1x2.home_win - self.current_1x2.home_win,
            delta_draw=new_1x2.draw - self.current_1x2.draw,
            delta_away_win=new_1x2.away_win - self.current_1x2.away_win,
            new_over_2_5=new_o25,
            delta_over_2_5=new_o25 - cur_o25,
        )
