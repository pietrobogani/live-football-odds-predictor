"""
Run the Poisson model on a specific goal from the validation data.

Usage:
    py run_goal.py          # list all goals + summary stats
    py run_goal.py 3        # run goal #3 directly
"""

import sys
import pandas as pd
from src.poisson_model import FootballPredictor, Probabilities

df = pd.read_csv('validation/results.csv')

# List all goals + summary
if len(sys.argv) < 2:
    print("Available goals:\n")
    for i, row in df.iterrows():
        print(f"  {i+1:2d}. {row['match']} | {row['minute']}' {row['player']} [{row['scorer'].upper()}] -> {row['score_after']}  (MAE: {row['mae']*100:.1f}%)")

    mae = df['mae']
    print(f"\n--- Validation Summary ({len(df)} goals, {df['match'].nunique()} matches) ---")
    print(f"Mean Absolute Error:   {mae.mean()*100:.1f}%")
    print(f"Median Absolute Error: {mae.median()*100:.1f}%")
    print(f"Within 5%:             {(mae <= 0.05).sum()}/{len(df)} = {(mae <= 0.05).mean()*100:.0f}%")
    print(f"Within 10%:            {(mae <= 0.10).sum()}/{len(df)} = {(mae <= 0.10).mean()*100:.0f}%")
    print(f"\nRun: py run_goal.py <number>")
    sys.exit(0)

idx = int(sys.argv[1]) - 1
row = df.iloc[idx]

# Parse score before goal
h_after, a_after = map(int, row['score_after'].split('-'))
if row['scorer'] == 'home':
    h_before, a_before = h_after - 1, a_after
else:
    h_before, a_before = h_after, a_after - 1

# Run model (time-based fallback since O/U prices not stored)
predictor = FootballPredictor()
predictor.update(
    minute=int(row['minute']),
    home_goals=h_before,
    away_goals=a_before,
    market_1x2=Probabilities(row['pre_home'], row['pre_draw'], row['pre_away']),
)

home_scores = row['scorer'] == 'home'
impact = predictor.predict_goal_impact(home_scores=home_scores)

print(f"\n{row['match']} | {row['minute']}' {row['player']} [{row['scorer'].upper()}]")
print(f"Score: {h_before}-{a_before} -> {row['score_after']}")
print(f"Calibration: {predictor.get_summary()['calibration_method']}")
print(f"lambda_home={predictor.get_summary()['lambda_home']}, lambda_away={predictor.get_summary()['lambda_away']}")
print()
print(f"{'':20s} {'Home':>10s} {'Draw':>10s} {'Away':>10s}")
print(f"{'Pre-goal market':20s} {row['pre_home']:10.1%} {row['pre_draw']:10.1%} {row['pre_away']:10.1%}")
print(f"{'Model prediction':20s} {impact.new_1x2.home_win:10.1%} {impact.new_1x2.draw:10.1%} {impact.new_1x2.away_win:10.1%}")
print(f"{'Actual post-goal':20s} {row['actual_home']:10.1%} {row['actual_draw']:10.1%} {row['actual_away']:10.1%}")
print(f"{'Error':20s} {impact.new_1x2.home_win - row['actual_home']:+10.1%} {impact.new_1x2.draw - row['actual_draw']:+10.1%} {impact.new_1x2.away_win - row['actual_away']:+10.1%}")
print(f"\nMAE: {row['mae']*100:.1f}%")
