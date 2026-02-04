"""
Poisson Model for Football Match Probability Prediction

Predicts how match probabilities change in real-time using Poisson/Skellam distributions.

Mathematical basis:
- Goals scored by each team follow independent Poisson distributions
- The difference of two Poissons follows a Skellam distribution
- We calibrate λ values from live market prices using dual calibration (O/U + 1X2)

Supported predictions:
- 1X2 (Home Win, Draw, Away Win)
- Over/Under (1.5, 2.5, 3.5, 4.5 goals)
- Handicap markets (±1.5)
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from scipy.special import iv as bessel_i
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize
import numpy as np
from enum import Enum


@dataclass
class MatchState:
    """Current state of a match."""
    minute: int
    home_goals: int
    away_goals: int

    @property
    def goal_diff(self) -> int:
        return self.home_goals - self.away_goals

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals


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
        return Probabilities(
            home_win=self.home_win / total,
            draw=self.draw / total,
            away_win=self.away_win / total
        )


@dataclass
class PoissonParams:
    """Poisson parameters for remaining goals."""
    lambda_home: float  # Expected remaining goals for home team
    lambda_away: float  # Expected remaining goals for away team

    @property
    def total(self) -> float:
        return self.lambda_home + self.lambda_away

    @property
    def home_share(self) -> float:
        if self.total == 0:
            return 0.5
        return self.lambda_home / self.total


# =============================================================================
# SKELLAM DISTRIBUTION
# =============================================================================

def skellam_pmf(k: int, lambda1: float, lambda2: float) -> float:
    """
    Probability mass function of the Skellam distribution.

    Skellam(k; λ₁, λ₂) = P(X - Y = k) where X ~ Poisson(λ₁), Y ~ Poisson(λ₂)

    Formula: e^{-(λ₁+λ₂)} × (λ₁/λ₂)^{k/2} × I_{|k|}(2√(λ₁λ₂))

    where I_n is the modified Bessel function of the first kind.
    """
    if lambda1 <= 0 or lambda2 <= 0:
        if lambda1 <= 0 and lambda2 <= 0:
            return 1.0 if k == 0 else 0.0
        elif lambda1 <= 0:
            if k > 0:
                return 0.0
            return math.exp(-lambda2) * (lambda2 ** (-k)) / math.factorial(-k) if k <= 0 else 0.0
        else:
            if k < 0:
                return 0.0
            return math.exp(-lambda1) * (lambda1 ** k) / math.factorial(k) if k >= 0 else 0.0

    sqrt_prod = math.sqrt(lambda1 * lambda2)
    ratio = lambda1 / lambda2
    log_prob = -(lambda1 + lambda2) + (k / 2) * math.log(ratio)
    bessel_val = bessel_i(abs(k), 2 * sqrt_prod)

    return math.exp(log_prob) * bessel_val


# =============================================================================
# PROBABILITY CALCULATIONS
# =============================================================================

def compute_1x2_from_poisson(lambda_home: float, lambda_away: float,
                              current_diff: int, max_goals: int = 15) -> Probabilities:
    """
    Compute 1X2 probabilities given Poisson parameters and current score.

    Uses Skellam distribution to sum over all possible remaining goal differences.
    """
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for additional_diff in range(-max_goals, max_goals + 1):
        prob = skellam_pmf(additional_diff, lambda_home, lambda_away)
        final_diff = current_diff + additional_diff

        if final_diff > 0:
            p_home_win += prob
        elif final_diff == 0:
            p_draw += prob
        else:
            p_away_win += prob

    return Probabilities(home_win=p_home_win, draw=p_draw, away_win=p_away_win)


def compute_over_under(lambda_home: float, lambda_away: float,
                       current_total: int, threshold: float) -> Tuple[float, float]:
    """
    Compute Over/Under probabilities.

    The sum of two independent Poissons is also Poisson:
    total_remaining ~ Poisson(λ_home + λ_away)
    """
    lambda_total = lambda_home + lambda_away
    goals_needed = max(0, math.ceil(threshold) - current_total)

    if goals_needed <= 0:
        return (1.0, 0.0)  # Already over

    if lambda_total <= 0.001:
        return (0.0, 1.0)  # No goals expected

    p_under = poisson_dist.cdf(goals_needed - 1, lambda_total)
    p_over = 1 - p_under

    return (p_over, p_under)


def compute_handicap(lambda_home: float, lambda_away: float,
                     current_diff: int, handicap: float, for_home: bool = True) -> float:
    """
    Compute handicap probability using Skellam distribution.

    Example: Home -1.5 means home needs to win by 2+ goals.
    """
    if for_home:
        threshold = math.ceil(-handicap)
    else:
        threshold = math.floor(handicap)

    prob = 0.0
    max_goals = 15

    for additional_diff in range(-max_goals, max_goals + 1):
        p = skellam_pmf(additional_diff, lambda_home, lambda_away)
        final_diff = current_diff + additional_diff

        if for_home:
            if final_diff >= threshold:
                prob += p
        else:
            if final_diff <= threshold:
                prob += p

    return prob


# =============================================================================
# DUAL CALIBRATION (O/U + 1X2)
# =============================================================================

class CalibrationMethod(Enum):
    DUAL = "dual"           # O/U + 1X2 (best accuracy)
    TIME_FALLBACK = "time"  # Time-based fallback
    FAILED = "failed"


@dataclass
class CalibrationResult:
    """Result of Poisson parameter calibration."""
    params: Optional[PoissonParams]
    method: CalibrationMethod
    confidence: float  # 0-1 reliability score
    m_total: Optional[float] = None  # Total remaining goals
    q_share: Optional[float] = None  # Home's share of goals


def solve_m_from_over_probability(target_over: float, goals_needed: int,
                                   m_min: float = 0.01, m_max: float = 10.0) -> float:
    """
    Solve for m (total remaining goals) given an Over probability.

    Uses binary search since P(Over) is monotonically increasing in m.
    """
    if goals_needed <= 0:
        raise ValueError(f"goals_needed must be positive, got {goals_needed}")

    if target_over <= 0.001:
        return m_min
    if target_over >= 0.999:
        return m_max

    low, high = m_min, m_max

    for _ in range(100):
        mid = (low + high) / 2
        p_over = 1 - poisson_dist.cdf(goals_needed - 1, mid)

        if abs(p_over - target_over) < 0.0001:
            return mid

        if p_over < target_over:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def calibrate_m_from_ou(ou_prices: Dict[float, float], current_goals: int,
                        min_informative: float = 0.15,
                        max_informative: float = 0.85) -> Tuple[Optional[float], float]:
    """
    Calibrate m (total remaining goals) from O/U market prices.

    Uses weighted average of implied m values, weighted by informativeness
    (prices closer to 50% are more informative).
    """
    if not ou_prices:
        return None, 0.0

    valid_lines = []

    for line, price in ou_prices.items():
        goals_needed = math.ceil(line) - current_goals

        if goals_needed <= 0:
            continue
        if price < min_informative or price > max_informative:
            continue

        try:
            m_implied = solve_m_from_over_probability(price, goals_needed)
            weight = 1 - 2 * abs(price - 0.50)
            weight = max(0.1, weight)

            valid_lines.append({
                'line': line,
                'm_implied': m_implied,
                'weight': weight
            })
        except Exception:
            continue

    if not valid_lines:
        return None, 0.0

    total_weight = sum(v['weight'] for v in valid_lines)
    m_weighted = sum(v['m_implied'] * v['weight'] for v in valid_lines) / total_weight

    # Confidence based on number of lines and consistency
    m_values = [v['m_implied'] for v in valid_lines]
    m_spread = max(m_values) - min(m_values)
    num_lines_factor = min(1.0, len(valid_lines) / 2)
    consistency_factor = max(0.3, 1 - m_spread / 3)

    confidence = num_lines_factor * consistency_factor
    confidence = max(0.3, min(0.95, confidence))

    return m_weighted, confidence


def calibrate_q_from_1x2(target_probs: Probabilities, m_total: float,
                         current_diff: int) -> Optional[float]:
    """
    Calibrate q (home goal share) from 1X2 prices given fixed m.

    Uses optimization to find q such that λ_home = m×q, λ_away = m×(1-q)
    produces the target 1X2 probabilities.
    """
    if m_total <= 0.001:
        return 0.5

    def error_fn(q_arr):
        q = q_arr[0]
        if q < 0.05 or q > 0.95:
            return 1e10

        lambda_home = m_total * q
        lambda_away = m_total * (1 - q)
        model_probs = compute_1x2_from_poisson(lambda_home, lambda_away, current_diff)

        return (
            (model_probs.home_win - target_probs.home_win) ** 2 +
            (model_probs.draw - target_probs.draw) ** 2 +
            (model_probs.away_win - target_probs.away_win) ** 2
        )

    initial_q = 0.55 if target_probs.home_win > target_probs.away_win else 0.45

    try:
        result = minimize(error_fn, x0=[initial_q], method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.001})
        return max(0.05, min(0.95, result.x[0]))
    except Exception:
        return None


def calibrate_poisson(target_1x2: Probabilities,
                      ou_prices: Optional[Dict[float, float]],
                      current_goals: int,
                      current_diff: int,
                      minutes_remaining: float) -> CalibrationResult:
    """
    Robust Poisson calibration with fallback hierarchy.

    Level 1: Dual calibration (O/U + 1X2) - best accuracy
    Level 2: Time-based fallback - uses expected goals from remaining time
    """
    # Level 1: Dual calibration
    if ou_prices:
        m, confidence = calibrate_m_from_ou(ou_prices, current_goals)

        if m is not None and 0.05 <= m <= 8.0:
            q = calibrate_q_from_1x2(target_1x2, m, current_diff)

            if q is not None and 0.05 <= q <= 0.95:
                return CalibrationResult(
                    params=PoissonParams(lambda_home=m * q, lambda_away=m * (1 - q)),
                    method=CalibrationMethod.DUAL,
                    confidence=confidence,
                    m_total=m,
                    q_share=q
                )

    # Level 2: Time-based fallback
    avg_goals_per_90 = 2.5
    time_fraction = minutes_remaining / 90.0
    m_estimated = avg_goals_per_90 * time_fraction

    # Estimate q from 1X2 direction
    if target_1x2.home_win > target_1x2.away_win:
        q_estimated = 0.55
    elif target_1x2.away_win > target_1x2.home_win:
        q_estimated = 0.45
    else:
        q_estimated = 0.50

    return CalibrationResult(
        params=PoissonParams(
            lambda_home=max(0.01, m_estimated * q_estimated),
            lambda_away=max(0.01, m_estimated * (1 - q_estimated))
        ),
        method=CalibrationMethod.TIME_FALLBACK,
        confidence=0.5,
        m_total=m_estimated,
        q_share=q_estimated
    )


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

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
    Main class for predicting probability changes in live football matches.

    Usage:
        predictor = FootballPredictor()
        predictor.update(
            minute=30,
            home_goals=0,
            away_goals=0,
            market_1x2=Probabilities(0.50, 0.28, 0.22),
            ou_prices={2.5: 0.55, 3.5: 0.30}
        )

        # What happens if home scores?
        impact = predictor.predict_goal_impact(home_scores=True)
        print(f"Home win probability: {impact.new_1x2.home_win:.1%}")
        print(f"Change: {impact.delta_home_win:+.1%}")
    """

    TOTAL_MATCH_DURATION = 96  # Minutes including stoppage time

    def __init__(self):
        self.state: Optional[MatchState] = None
        self.current_1x2: Optional[Probabilities] = None
        self.params: Optional[PoissonParams] = None
        self.calibration: Optional[CalibrationResult] = None

    def update(self, minute: int, home_goals: int, away_goals: int,
               market_1x2: Probabilities,
               ou_prices: Optional[Dict[float, float]] = None) -> None:
        """
        Update model with current match state and market prices.

        Args:
            minute: Current match minute (0-96 scale)
            home_goals: Current home team goals
            away_goals: Current away team goals
            market_1x2: Current 1X2 probabilities from market
            ou_prices: Optional O/U prices {line: over_probability}
        """
        self.state = MatchState(minute=minute, home_goals=home_goals, away_goals=away_goals)
        self.current_1x2 = market_1x2.normalize()

        minutes_remaining = max(0, self.TOTAL_MATCH_DURATION - minute)

        self.calibration = calibrate_poisson(
            target_1x2=self.current_1x2,
            ou_prices=ou_prices,
            current_goals=self.state.total_goals,
            current_diff=self.state.goal_diff,
            minutes_remaining=minutes_remaining
        )

        self.params = self.calibration.params

    def predict_goal_impact(self, home_scores: bool) -> GoalImpact:
        """
        Predict how probabilities change if a goal is scored.

        Args:
            home_scores: True if home team scores, False if away

        Returns:
            GoalImpact with new probabilities and changes
        """
        if self.state is None or self.params is None or self.current_1x2 is None:
            raise ValueError("Must call update() before predict_goal_impact()")

        new_diff = self.state.goal_diff + (1 if home_scores else -1)
        new_total = self.state.total_goals + 1

        new_1x2 = compute_1x2_from_poisson(
            self.params.lambda_home,
            self.params.lambda_away,
            new_diff
        )

        current_over_2_5, _ = compute_over_under(
            self.params.lambda_home,
            self.params.lambda_away,
            self.state.total_goals,
            threshold=2.5
        )

        new_over_2_5, _ = compute_over_under(
            self.params.lambda_home,
            self.params.lambda_away,
            new_total,
            threshold=2.5
        )

        return GoalImpact(
            new_1x2=new_1x2,
            delta_home_win=new_1x2.home_win - self.current_1x2.home_win,
            delta_draw=new_1x2.draw - self.current_1x2.draw,
            delta_away_win=new_1x2.away_win - self.current_1x2.away_win,
            new_over_2_5=new_over_2_5,
            delta_over_2_5=new_over_2_5 - current_over_2_5
        )

    def get_summary(self) -> dict:
        """Get current model state summary."""
        if self.state is None:
            return {'initialized': False}

        return {
            'minute': self.state.minute,
            'score': f"{self.state.home_goals}-{self.state.away_goals}",
            'calibration_method': self.calibration.method.value if self.calibration else None,
            'confidence': round(self.calibration.confidence, 2) if self.calibration else None,
            'lambda_home': round(self.params.lambda_home, 3) if self.params else None,
            'lambda_away': round(self.params.lambda_away, 3) if self.params else None,
            'current_1x2': {
                'home': round(self.current_1x2.home_win, 3),
                'draw': round(self.current_1x2.draw, 3),
                'away': round(self.current_1x2.away_win, 3)
            } if self.current_1x2 else None
        }


if __name__ == "__main__":
    # Example: 0-0 at 30 minutes, home slightly favored
    predictor = FootballPredictor()
    predictor.update(
        minute=30,
        home_goals=0,
        away_goals=0,
        market_1x2=Probabilities(home_win=0.50, draw=0.28, away_win=0.22),
        ou_prices={2.5: 0.55, 3.5: 0.30}
    )

    print("Current state:")
    print(predictor.get_summary())

    print("\nIf HOME scores:")
    impact = predictor.predict_goal_impact(home_scores=True)
    print(f"  Home win: {impact.new_1x2.home_win:.1%} ({impact.delta_home_win:+.1%})")
    print(f"  Draw:     {impact.new_1x2.draw:.1%} ({impact.delta_draw:+.1%})")
    print(f"  Away win: {impact.new_1x2.away_win:.1%} ({impact.delta_away_win:+.1%})")

    print("\nIf AWAY scores:")
    impact = predictor.predict_goal_impact(home_scores=False)
    print(f"  Home win: {impact.new_1x2.home_win:.1%} ({impact.delta_home_win:+.1%})")
    print(f"  Draw:     {impact.new_1x2.draw:.1%} ({impact.delta_draw:+.1%})")
    print(f"  Away win: {impact.new_1x2.away_win:.1%} ({impact.delta_away_win:+.1%})")
