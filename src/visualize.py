"""
Dark mode visualization for football probability charts.

Style characteristics:
- Smooth probability lines
- Y-axis with percentages
- Minimal design with subtle gridlines
- Outcome-specific color palette
- End dot marker at current value
- Dark background
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Dark mode color palette
COLORS = {
    'primary': '#60A5FA',      # Bright blue
    'secondary': '#93C5FD',    # Lighter blue
    'positive': '#4ADE80',     # Bright green
    'negative': '#F87171',     # Bright red
    'neutral': '#9CA3AF',      # Light gray
    'grid': '#374151',         # Dark gray grid
    'text': '#F3F4F6',         # Light text
    'background': '#111827',   # Dark background
    'goal_marker': '#A78BFA',  # Purple for goal events
    'prediction': '#34D399',   # Green for predictions (before goal)
    'prediction_locked': '#A78BFA',  # Purple for locked predictions (after goal)
}

# Team-specific colors (for 1X2 markets) - glowing for dark mode
OUTCOME_COLORS = {
    'home': '#38BDF8',    # Sky blue (glowing)
    'draw': '#FCD34D',    # Yellow/amber (glowing)
    'away': '#FB7185',    # Rose/red (glowing)
    'over': '#4ADE80',    # Bright green
    'under': '#FB7185',   # Rose/red
}


def setup_style():
    """Configure matplotlib for dark mode plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'medium',
        'axes.titlecolor': COLORS['text'],
        'axes.labelsize': 10,
        'axes.labelcolor': COLORS['neutral'],
        'axes.edgecolor': COLORS['grid'],
        'axes.linewidth': 0.5,
        'axes.facecolor': COLORS['background'],
        'figure.facecolor': COLORS['background'],
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 1.0,
        'xtick.color': COLORS['neutral'],
        'ytick.color': COLORS['neutral'],
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'legend.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
    })


def create_probability_line(ax, x, y, color=None, label=None, show_end_dot=True):
    """
    Create a probability line on the given axes.

    Args:
        ax: matplotlib axes
        x: x-axis values (e.g., minutes)
        y: y-axis values (probabilities 0-1)
        color: line color (default: primary blue)
        label: optional label for legend
        show_end_dot: show dot at final value
    """
    color = color or COLORS['primary']

    # Smooth line
    ax.plot(x, y, color=color, linewidth=2, label=label)

    # End dot marker
    if show_end_dot and len(x) > 0:
        ax.scatter([x[-1]], [y[-1]], color=color, s=40, zorder=5, edgecolors='white', linewidths=1.5)


def style_axes(ax, title=None, y_range=(0, 1), show_grid=True):
    """
    Apply styling to axes.

    Args:
        ax: matplotlib axes
        title: optional title
        y_range: tuple of (min, max) for y-axis
        show_grid: whether to show horizontal gridlines
    """
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.spines['left'].set_color(COLORS['grid'])

    # Y-axis on left (default)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    # Y-axis as percentage
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(y_range)

    # Horizontal gridlines only
    if show_grid:
        ax.grid(True, axis='y', linestyle='-', linewidth=0.5, color=COLORS['grid'])
        ax.grid(False, axis='x')
    else:
        ax.grid(False)

    # Title
    if title:
        ax.set_title(title, loc='left', fontweight='medium', color=COLORS['text'])


def add_goal_marker(ax, x, label=None, color=None):
    """
    Add a vertical line marking a goal event.

    Args:
        ax: matplotlib axes
        x: x-position of the goal
        label: optional label (e.g., "⚽ 32'")
        color: marker color
    """
    color = color or COLORS['goal_marker']
    ax.axvline(x=x, color=color, linewidth=1.5, linestyle='--', alpha=0.7)

    if label:
        ax.annotate(label, xy=(x, ax.get_ylim()[1]), xytext=(3, -5),
                    textcoords='offset points', fontsize=9, color=color,
                    fontweight='medium', va='top')


def add_prediction_line(ax, x, y, goal_minute=None, color=None, label='Prediction'):
    """
    Add a prediction line (dotted). Changes color at goal minute.

    Args:
        ax: matplotlib axes
        x: x-axis values
        y: y-axis values
        goal_minute: minute when goal occurred (for color split)
        color: line color (used if no goal_minute)
        label: legend label
    """
    if goal_minute is None:
        color = color or COLORS['prediction']
        ax.plot(x, y, color=color, linewidth=2, linestyle=':',
                alpha=0.9, label=label)
    else:
        # Split into before/after goal
        before_x, before_y = [], []
        after_x, after_y = [], []

        for xi, yi in zip(x, y):
            if xi <= goal_minute:
                before_x.append(xi)
                before_y.append(yi)
            else:
                after_x.append(xi)
                after_y.append(yi)

        # Draw before goal (green)
        if before_x:
            ax.plot(before_x, before_y, color=COLORS['prediction'],
                    linewidth=2, linestyle=':', alpha=0.9)

        # Draw after goal (purple) - include connection point
        if after_x:
            # Add last pre-goal point to connect the lines
            if before_x:
                after_x = [before_x[-1]] + after_x
                after_y = [before_y[-1]] + after_y
            ax.plot(after_x, after_y, color=COLORS['prediction_locked'],
                    linewidth=2, linestyle=':', alpha=0.9)


def create_comparison_chart(minutes, observed, predicted, goal_minute,
                            title='', outcome_type='home', figsize=(10, 4)):
    """
    Create a single chart comparing observed vs predicted probabilities.

    Args:
        minutes: list of minute values
        observed: list of observed probabilities
        predicted: list of predicted probabilities
        goal_minute: minute when goal occurred
        title: chart title
        outcome_type: 'home', 'draw', 'away', 'over', 'under'
        figsize: figure size tuple

    Returns:
        fig, ax tuple
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    color = OUTCOME_COLORS.get(outcome_type, COLORS['primary'])

    # Observed probabilities
    create_probability_line(ax, minutes, observed, color=color, label='Market')

    # Model predictions (filter None values)
    valid_idx = [i for i, v in enumerate(predicted) if v is not None]
    if valid_idx:
        pred_mins = [minutes[i] for i in valid_idx]
        pred_vals = [predicted[i] for i in valid_idx]
        add_prediction_line(ax, pred_mins, pred_vals, label='Poisson')

    # Goal marker
    add_goal_marker(ax, goal_minute, label=f"GOAL {goal_minute}'")

    # Style
    style_axes(ax, title=title)
    ax.set_xlabel('Minute', color=COLORS['neutral'])

    # Legend (minimal, top-left)
    ax.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    return fig, ax


def create_triple_chart(trajectory, goal_info, match_info, figsize=(16, 4)):
    """
    Create a 3-panel chart for Home/Draw/Away outcomes.

    Args:
        trajectory: dict with minutes, obs_*, pred_* keys
        goal_info: dict with minute, player, scorer, score_after
        match_info: dict with home_team, away_team

    Returns:
        fig, axes tuple
    """
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    minutes = trajectory['minutes']
    goal_minute = goal_info['minute']

    panels = [
        ('Home Win', 'obs_home', 'pred_home', 'home'),
        ('Draw', 'obs_draw', 'pred_draw', 'draw'),
        ('Away Win', 'obs_away', 'pred_away', 'away'),
    ]

    for ax, (title, obs_key, pred_key, outcome_type) in zip(axes, panels):
        color = OUTCOME_COLORS[outcome_type]

        # Observed probabilities
        create_probability_line(ax, minutes, trajectory[obs_key], color=color, label='Market')

        # Model predictions (green before goal, purple after)
        pred_vals = trajectory[pred_key]
        valid_idx = [i for i, v in enumerate(pred_vals) if v is not None]
        if valid_idx:
            pred_mins = [minutes[i] for i in valid_idx]
            pred_vals_clean = [pred_vals[i] for i in valid_idx]
            add_prediction_line(ax, pred_mins, pred_vals_clean, goal_minute=goal_minute)

        # Goal marker
        add_goal_marker(ax, goal_minute)

        # Style
        style_axes(ax, title=title)
        ax.set_xlabel('Minute', color=COLORS['neutral'])


    # Super title
    scorer_label = '[H]' if goal_info['scorer'] == 'home' else '[A]'
    fig.suptitle(
        f"{match_info['home_team']} vs {match_info['away_team']}  |  "
        f"{scorer_label} {goal_info['minute']}' {goal_info['player']}  |  "
        f"{goal_info['score_after']}",
        fontsize=12, fontweight='medium', color=COLORS['text'], y=1.02
    )

    plt.tight_layout()
    return fig, axes


def create_single_outcome_chart(trajectory, goal_info, match_info,
                                 outcome='home', figsize=(10, 5)):
    """
    Create a detailed single-outcome chart with more visual polish.

    Args:
        trajectory: dict with minutes, obs_*, pred_* keys
        goal_info: dict with minute, player, scorer, score_after
        match_info: dict with home_team, away_team
        outcome: 'home', 'draw', or 'away'

    Returns:
        fig, ax tuple
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    minutes = trajectory['minutes']
    goal_minute = goal_info['minute']

    obs_key = f'obs_{outcome}'
    pred_key = f'pred_{outcome}'

    observed = trajectory[obs_key]
    predicted = trajectory[pred_key]
    color = OUTCOME_COLORS[outcome]

    # Fill area under observed line
    ax.fill_between(minutes, 0, observed, alpha=0.1, color=color)

    # Observed probabilities
    create_probability_line(ax, minutes, observed, color=color, label='Market')

    # Model predictions
    valid_idx = [i for i, v in enumerate(predicted) if v is not None]
    if valid_idx:
        pred_mins = [minutes[i] for i in valid_idx]
        pred_vals = [predicted[i] for i in valid_idx]
        add_prediction_line(ax, pred_mins, pred_vals, label='Poisson Prediction')

    # Goal marker with shading
    ax.axvspan(goal_minute - 0.3, goal_minute + 0.3, alpha=0.15,
               color=COLORS['goal_marker'], zorder=0)
    add_goal_marker(ax, goal_minute, label=f"GOAL: {goal_info['player']}")

    # Style
    outcome_names = {'home': 'Home Win', 'draw': 'Draw', 'away': 'Away Win'}
    style_axes(ax, title=outcome_names[outcome])
    ax.set_xlabel('Minute', color=COLORS['neutral'])

    # Legend
    ax.legend(loc='upper left', frameon=False)

    # Match info annotation
    scorer_label = '[H]' if goal_info['scorer'] == 'home' else '[A]'
    info_text = (f"{match_info['home_team']} vs {match_info['away_team']}\n"
                 f"{scorer_label} {goal_info['minute']}' - {goal_info['score_after']}")
    ax.annotate(info_text, xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=9, color=COLORS['text'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['background'],
                         edgecolor=COLORS['grid'], alpha=0.9))

    plt.tight_layout()
    return fig, ax
