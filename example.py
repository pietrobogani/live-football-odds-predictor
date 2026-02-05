"""
Example: Predict how probabilities change when a goal is scored.

Run: py example.py
"""

from src.poisson_model import FootballPredictor, Probabilities

predictor = FootballPredictor()

# Match state: 0-0 at 30 minutes, home slightly favored
# Market 1X2 probabilities and Over/Under prices from a bookmaker
predictor.update(
    minute=30,
    home_goals=0,
    away_goals=0,
    market_1x2=Probabilities(home_win=0.50, draw=0.28, away_win=0.22),
    ou_prices={2.5: 0.55, 3.5: 0.30}
)

print(f"Score: 0-0  Minute: 30'")
print(f"Calibrated: lambda_home={predictor.lambda_home:.3f}, lambda_away={predictor.lambda_away:.3f}")

# What happens if home scores?
home_goal = predictor.predict_goal_impact(home_scores=True)
print(f"\nIf HOME scores (1-0):")
print(f"  Home win: {home_goal.new_1x2.home_win:.1%} ({home_goal.delta_home_win:+.1%})")
print(f"  Draw:     {home_goal.new_1x2.draw:.1%} ({home_goal.delta_draw:+.1%})")
print(f"  Away win: {home_goal.new_1x2.away_win:.1%} ({home_goal.delta_away_win:+.1%})")
print(f"  Over 2.5: {home_goal.new_over_2_5:.1%} ({home_goal.delta_over_2_5:+.1%})")

# What happens if away scores?
away_goal = predictor.predict_goal_impact(home_scores=False)
print(f"\nIf AWAY scores (0-1):")
print(f"  Home win: {away_goal.new_1x2.home_win:.1%} ({away_goal.delta_home_win:+.1%})")
print(f"  Draw:     {away_goal.new_1x2.draw:.1%} ({away_goal.delta_draw:+.1%})")
print(f"  Away win: {away_goal.new_1x2.away_win:.1%} ({away_goal.delta_away_win:+.1%})")
print(f"  Over 2.5: {away_goal.new_over_2_5:.1%} ({away_goal.delta_over_2_5:+.1%})")
