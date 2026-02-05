"""Plotting utilities for validation charts (white background, minimal style)."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLORS = {
    'neutral': '#6B7280',
    'grid': '#E5E7EB',
    'text': '#111827',
    'background': '#FFFFFF',
    'goal_marker': '#7C3AED',       # Purple
    'prediction': '#059669',         # Green (before goal)
    'prediction_locked': '#7C3AED',  # Purple (after goal, "locked")
}

OUTCOME_COLORS = {'home': '#2563EB', 'draw': '#D97706', 'away': '#DC2626'}


def setup_style():
    """Configure matplotlib rcParams."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Arial', 'Helvetica'],
        'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'medium',
        'axes.titlecolor': COLORS['text'], 'axes.labelsize': 10,
        'axes.labelcolor': COLORS['neutral'], 'axes.edgecolor': COLORS['grid'],
        'axes.linewidth': 0.5, 'axes.facecolor': COLORS['background'],
        'figure.facecolor': COLORS['background'],
        'grid.color': COLORS['grid'], 'grid.linewidth': 0.5, 'grid.alpha': 1.0,
        'xtick.color': COLORS['neutral'], 'ytick.color': COLORS['neutral'],
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False,
        'legend.labelcolor': COLORS['text'], 'text.color': COLORS['text'],
    })


def _style_axes(ax, title=None):
    """Remove top/right spines, percentage y-axis, horizontal grid only."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.spines['left'].set_color(COLORS['grid'])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, axis='y', linestyle='-', linewidth=0.5, color=COLORS['grid'])
    ax.grid(False, axis='x')
    if title:
        ax.set_title(title, loc='left', fontweight='medium', color=COLORS['text'])


def _plot_observed(ax, x, y, color):
    """Solid line with end-dot for observed market probabilities."""
    ax.plot(x, y, color=color, linewidth=2, label='Market')
    if len(x) > 0:
        ax.scatter([x[-1]], [y[-1]], color=color, s=40, zorder=5, edgecolors='white', linewidths=1.5)


def _plot_prediction(ax, x, y, goal_minute):
    """Dotted line: green before goal, purple after (locked prediction)."""
    before_x, before_y, after_x, after_y = [], [], [], []
    for xi, yi in zip(x, y):
        (before_x if xi <= goal_minute else after_x).append(xi)
        (before_y if xi <= goal_minute else after_y).append(yi)
    if before_x:
        ax.plot(before_x, before_y, color=COLORS['prediction'], linewidth=2, linestyle=':', alpha=0.9)
    if after_x:
        if before_x:  # connect at goal minute
            after_x = [before_x[-1]] + after_x
            after_y = [before_y[-1]] + after_y
        ax.plot(after_x, after_y, color=COLORS['prediction_locked'], linewidth=2, linestyle=':', alpha=0.9)


def create_triple_chart(trajectory, goal_info, match_info, figsize=(16, 4)):
    """
    3-panel chart (Home/Draw/Away) comparing market vs model predictions.

    trajectory: dict with keys minutes, obs_home/draw/away, pred_home/draw/away
    goal_info: dict with minute, player, scorer, score_after
    match_info: dict with home_team, away_team
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    minutes = trajectory['minutes']
    goal_minute = goal_info['minute']

    for ax, (title, obs_key, pred_key, otype) in zip(axes, [
        ('Home Win', 'obs_home', 'pred_home', 'home'),
        ('Draw', 'obs_draw', 'pred_draw', 'draw'),
        ('Away Win', 'obs_away', 'pred_away', 'away'),
    ]):
        _plot_observed(ax, minutes, trajectory[obs_key], OUTCOME_COLORS[otype])

        pred = trajectory[pred_key]
        valid = [(minutes[i], pred[i]) for i in range(len(pred)) if pred[i] is not None]
        if valid:
            _plot_prediction(ax, [v[0] for v in valid], [v[1] for v in valid], goal_minute)

        ax.axvline(x=goal_minute, color=COLORS['goal_marker'], linewidth=1.5, linestyle='--', alpha=0.7)
        _style_axes(ax, title=title)
        ax.set_xlabel('Minute', color=COLORS['neutral'])

    scorer_label = '[H]' if goal_info['scorer'] == 'home' else '[A]'
    fig.suptitle(
        f"{match_info['home_team']} vs {match_info['away_team']}  |  "
        f"{scorer_label} {goal_info['minute']}' {goal_info['player']}  |  "
        f"{goal_info['score_after']}",
        fontsize=12, fontweight='medium', color=COLORS['text'], y=1.02
    )
    plt.tight_layout()
    return fig, axes
