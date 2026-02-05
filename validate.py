"""
Validate the Poisson model against real match data.

Two modes:
    py validate.py              # reproduce MAE stats from results.csv
    py validate.py plot         # generate orderbook plots for all goals
    py validate.py plot [N]     # generate plot for goal N only
    py validate.py [N]          # show detailed prediction for goal N
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
from src.poisson_model import FootballPredictor, Probabilities
from src.visualize import setup_style, create_triple_chart, COLORS

RESULTS = Path('validation/results.csv')
ORDERBOOK_DIR = Path('validation/orderbooks')
OUTPUT_DIR = Path('validation/plots')
WINDOW = 6  # minutes before/after goal to show in plot

# Map match names in results.csv -> orderbook log filenames
MATCH_TO_LOG = {
    'Inter vs Arsenal': 'orderbook_2026-01-20_FC_Internazionale_Milano_vs._Arsenal_FC',
    'FC Kobenhavn vs Napoli': 'orderbook_2026-01-20_FC_København_vs._SSC_Napoli',
    'Sporting CP vs Paris Saint-Germain': 'orderbook_2026-01-20_Sporting_CP_vs._Paris_Saint_Germain_FC',
    'Udinese vs Roma': 'orderbook_2026-02-02_Udinese_Calcio_vs._AS_Roma',
    'Mallorca vs Sevilla': 'orderbook_2026-02-02_RCD_Mallorca_vs._Sevilla_FC',
    'Universitatea Cluj vs FC Arges Pitesti': 'orderbook_2026-02-03_FC_Universitatea_Cluj_vs._FC_Argeș_Pitești',
    'Al Ettifaq vs Al Taawoun': 'orderbook_2026-02-03_Al_Ettifaq_Saudi_Club_vs._Al_Taawoun_Saudi_Club',
    'Al Khaleej vs Al Qadisiyah': 'orderbook_2026-02-03_Al_Khaleej_Saudi_Club_vs._Al_Qadisiyah_Saudi_Club',
    'Bologna vs Milan': 'orderbook_2026-02-03_Bologna_FC_1909_vs._AC_Milan',
    'Hermannstadt vs Rapid Bucuresti': 'orderbook_2026-02-03_FC_Hermannstadt_vs._FC_Rapid_1923',
}

# Map match names -> substrings to identify home/away WIN markets in orderbook
MATCH_TO_MARKETS = {
    'Inter vs Arsenal': ('INTERNAZIONALE', 'ARSENAL'),
    'FC Kobenhavn vs Napoli': ('KØBENHAVN', 'NAPOLI'),
    'Sporting CP vs Paris Saint-Germain': ('SPORTING', 'PARIS'),
    'Udinese vs Roma': ('UDINESE', 'ROMA'),
    'Mallorca vs Sevilla': ('MALLORCA', 'SEVILLA'),
    'Universitatea Cluj vs FC Arges Pitesti': ('UNIVERSITATEA', 'ARGEȘ'),
    'Al Ettifaq vs Al Taawoun': ('ETTIFAQ', 'TAAWOUN'),
    'Al Khaleej vs Al Qadisiyah': ('KHALEEJ', 'QADISIYAH'),
    'Bologna vs Milan': ('BOLOGNA', 'MILAN'),
    'Hermannstadt vs Rapid Bucuresti': ('HERMANNSTADT', 'RAPID'),
}


def score_before_goal(row):
    """Derive score BEFORE the goal from score_after."""
    home, away = map(int, row['score_after'].split('-'))
    if row['scorer'] == 'home': home -= 1
    else: away -= 1
    return home, away


def run_prediction(r):
    """Re-run the Poisson model on a goal and return predicted post-goal 1X2."""
    home, away = score_before_goal(r)
    ou = {}
    for line in [1.5, 2.5, 3.5, 4.5]:
        col = f'ou_{line}'.replace('.', '_')
        if pd.notna(r.get(col)):
            ou[line] = r[col]
    predictor = FootballPredictor()
    predictor.update(r['minute'], home, away,
                     Probabilities(r['pre_home'], r['pre_draw'], r['pre_away']), ou)
    return predictor.predict_goal_impact(home_scores=(r['scorer'] == 'home')).new_1x2


# -- MAE stats -----------------------------------------------------------------

def print_mae_all(df):
    """Print per-goal MAE and summary statistics."""
    maes = []
    for i, r in df.iterrows():
        pred = run_prediction(r)
        mae = (abs(pred.home_win - r['actual_home']) + abs(pred.draw - r['actual_draw']) + abs(pred.away_win - r['actual_away'])) / 3
        maes.append(mae)
        print(f"  {i+1:2d}. {r['match']} | {r['minute']}' {r['player']} [{r['scorer'].upper()}] -> {r['score_after']}  (MAE: {mae*100:.1f}%)")
    maes = pd.Series(maes)
    print(f"\n--- {len(df)} goals, {df['match'].nunique()} matches ---")
    print(f"MAE: {maes.mean()*100:.1f}% | Median: {maes.median()*100:.1f}% | Within 5%: {(maes<=.05).mean()*100:.0f}% | Within 10%: {(maes<=.10).mean()*100:.0f}%")


def print_mae_single(r):
    """Print detailed prediction breakdown for a single goal."""
    pred = run_prediction(r)
    err_h = pred.home_win - r['actual_home']
    err_d = pred.draw - r['actual_draw']
    err_a = pred.away_win - r['actual_away']
    mae = (abs(err_h) + abs(err_d) + abs(err_a)) / 3
    print(f"\n{r['match']} | {r['minute']}' {r['player']} [{r['scorer'].upper()}] -> {r['score_after']}")
    print(f"\n{'':20s} {'Home':>10s} {'Draw':>10s} {'Away':>10s}")
    print(f"{'Pre-goal market':20s} {r['pre_home']:10.1%} {r['pre_draw']:10.1%} {r['pre_away']:10.1%}")
    print(f"{'Model prediction':20s} {pred.home_win:10.1%} {pred.draw:10.1%} {pred.away_win:10.1%}")
    print(f"{'Actual post-goal':20s} {r['actual_home']:10.1%} {r['actual_draw']:10.1%} {r['actual_away']:10.1%}")
    print(f"{'Error':20s} {err_h:+10.1%} {err_d:+10.1%} {err_a:+10.1%}")
    print(f"\nMAE: {mae*100:.1f}%")


# -- Plotting ------------------------------------------------------------------

def load_orderbook(filepath, home_key, away_key):
    """
    Load an orderbook CSV and extract per-minute midpoint prices.

    Returns a DataFrame with columns: minute, home, away, draw, ou_1.5, ou_2.5, ...
    Midpoint = (best_bid + best_ask) / 2 for each market.
    """
    with open(filepath) as f:
        n_cols = len(f.readline().strip().split(','))
    df = pd.read_csv(filepath, on_bad_lines=lambda cols: cols[:n_cols], engine='python')
    df['mid'] = (df['best_bid'] + df['best_ask']) / 2

    rows = []
    for minute, group in df.groupby('minute'):
        r = {'minute': int(minute)}
        for _, row in group.iterrows():
            market = row['market'].upper()
            if home_key in market and 'WIN' in market:
                r['home'] = row['mid']
            elif away_key in market and 'WIN' in market:
                r['away'] = row['mid']
            elif 'OVER' in market:
                line = float(market.replace('OVER ', ''))
                r[f'ou_{line}'] = row['mid']
        if 'home' in r and 'away' in r:
            r['draw'] = max(0, 1 - r['home'] - r['away'])
            rows.append(r)
    return pd.DataFrame(rows).sort_values('minute')


def get_prices_at(prices_df, minute):
    """Get the most recent prices at or before a given minute."""
    before = prices_df[prices_df['minute'] <= minute]
    return before.iloc[-1] if not before.empty else None


def compute_trajectory(prices_df, goal_minute, scorer, home_goals, away_goals):
    """
    Compute model predictions in a window around the goal.

    For each minute BEFORE the goal: run the model with that minute's market
    prices to predict what probabilities would look like after a goal.
    Uses T-1 prices (one minute prior) to avoid look-ahead bias.

    At and after the goal: the prediction "locks" — it stays at whatever the
    model predicted right before the goal. This lets us visually compare the
    locked prediction (dotted line) against where the market actually moved
    (solid line) after the goal.
    """
    start = max(1, goal_minute - WINDOW)
    end = goal_minute + WINDOW
    home_scores = scorer == 'home'

    # Compute the locked prediction using prices from 1 minute before the goal
    pre = get_prices_at(prices_df[prices_df['minute'] < goal_minute], goal_minute - 1)
    locked = None
    if pre is not None:
        ou = {k: pre[f'ou_{k}'] for k in [1.5, 2.5, 3.5, 4.5] if f'ou_{k}' in pre and pd.notna(pre[f'ou_{k}'])}
        predictor = FootballPredictor()
        predictor.update(goal_minute, home_goals, away_goals,
                         Probabilities(pre['home'], pre['draw'], pre['away']), ou)
        impact = predictor.predict_goal_impact(home_scores)
        locked = (impact.new_1x2.home_win, impact.new_1x2.draw, impact.new_1x2.away_win)

    mins, obs_h, obs_d, obs_a = [], [], [], []
    pred_h, pred_d, pred_a = [], [], []

    for m in range(start, end + 1):
        obs = get_prices_at(prices_df[prices_df['minute'] <= m], m)
        if obs is None:
            continue
        mins.append(m)
        obs_h.append(obs['home']); obs_d.append(obs['draw']); obs_a.append(obs['away'])

        if m < goal_minute:
            prev = get_prices_at(prices_df[prices_df['minute'] < m], m - 1)
            if prev is None:
                prev = obs
            ou = {k: prev[f'ou_{k}'] for k in [1.5, 2.5, 3.5, 4.5] if f'ou_{k}' in prev and pd.notna(prev[f'ou_{k}'])}
            try:
                predictor = FootballPredictor()
                predictor.update(m, home_goals, away_goals,
                                 Probabilities(prev['home'], prev['draw'], prev['away']), ou)
                impact = predictor.predict_goal_impact(home_scores)
                pred_h.append(impact.new_1x2.home_win)
                pred_d.append(impact.new_1x2.draw)
                pred_a.append(impact.new_1x2.away_win)
            except:
                pred_h.append(None); pred_d.append(None); pred_a.append(None)
        else:
            pred_h.append(locked[0] if locked else None)
            pred_d.append(locked[1] if locked else None)
            pred_a.append(locked[2] if locked else None)

    return {
        'minutes': mins,
        'obs_home': obs_h, 'obs_draw': obs_d, 'obs_away': obs_a,
        'pred_home': pred_h, 'pred_draw': pred_d, 'pred_away': pred_a,
        'goal_minute': goal_minute,
    }


def plot_goal(row, save=True):
    """Generate a 3-panel plot for a single goal."""
    match = row['match']
    log_file = ORDERBOOK_DIR / f"{MATCH_TO_LOG[match]}.csv"
    home_key, away_key = MATCH_TO_MARKETS[match]
    home, away = score_before_goal(row)

    prices = load_orderbook(log_file, home_key, away_key)
    traj = compute_trajectory(prices, row['minute'], row['scorer'], home, away)

    if not traj['minutes']:
        print(f"  No data for {match} {row['minute']}'"); return
    if np.var(traj['obs_home']) + np.var(traj['obs_draw']) + np.var(traj['obs_away']) < 1e-6:
        print(f"  Flat data for {match} {row['minute']}'"); return

    home_team, away_team = match.split(' vs ')
    goal_info = {'minute': row['minute'], 'player': row['player'],
                 'scorer': row['scorer'], 'score_after': row['score_after']}

    setup_style()
    fig, axes = create_triple_chart(traj, goal_info, {'home_team': home_team, 'away_team': away_team})

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe = MATCH_TO_LOG[match].replace('.', '_')
        out = OUTPUT_DIR / f"{safe}_goal_{row['minute']}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'], edgecolor='none')
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Saved: {str(out).encode('ascii', 'replace').decode()}")
    return fig


# -- CLI -----------------------------------------------------------------------

if __name__ == '__main__':
    df = pd.read_csv(RESULTS)
    args = sys.argv[1:]

    if not args:
        # py validate.py -> print MAE for all goals
        print_mae_all(df)
    elif args[0] == 'plot':
        # py validate.py plot [N] -> generate plots
        if len(args) >= 2:
            row = df.iloc[int(args[1]) - 1]
            print(f"{row['match']} | {row['minute']}' {row['player']} [{row['scorer'].upper()}] -> {row['score_after']}")
            plot_goal(row)
        else:
            print(f"Generating plots for {len(df)} goals...\n")
            for i, row in df.iterrows():
                print(f"{i+1:2d}. {row['match']} | {row['minute']}' {row['player']}")
                plot_goal(row)
            print(f"\nPlots saved to: {OUTPUT_DIR}")
    else:
        # py validate.py N -> detailed single goal
        print_mae_single(df.iloc[int(args[0]) - 1])
