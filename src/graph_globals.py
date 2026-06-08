import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def global_params():
    """Set publication-quality matplotlib defaults for dissertation figures."""
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'mathtext.fontset': 'stix',
        # Axes
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.titlepad': 8,
        'axes.labelpad': 6,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.grid.which': 'major',
        # Ticks
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        # Legend
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        # Figure
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'figure.figsize': (8, 5),
        # Grid
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'grid.linewidth': 0.5,
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        # Save
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# Consistent colour palette for all TSC controllers
TSC_COLOURS = {
    'ddpg':     '#2196F3',   # blue
    'dqn':      '#4CAF50',   # green
    'ppo':      '#FF9800',   # orange
    'sotl':     '#9C27B0',   # purple
    'websters': '#F44336',   # red
}

TSC_LABELS = {
    'ddpg':     'DDPG',
    'dqn':      'DQN',
    'ppo':      'PPO',
    'sotl':     'SOTL',
    'websters': "Webster's",
}
