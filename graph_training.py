import os, argparse

import numpy as np
import matplotlib.pyplot as plt

from src.graph_globals import global_params, TSC_COLOURS
from src.picklefuncs import load_data


def smooth(data, window=50):
    """Exponential moving average for noisy training curves."""
    if len(data) < window:
        return data
    alpha = 2 / (window + 1)
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def get_headers(fp):
    with open(fp, 'r') as f:
        header = f.readline()
        headers = header.split(',')
        headers[-1] = headers[-1].strip()
    # skip first column (timestamp)
    headers = headers[1:]
    return headers

def get_data(fp):
    with open(fp, 'r') as f:
        ncols = len(f.readline().split(','))
    data = np.loadtxt(fp, delimiter=',', skiprows=1, usecols=range(1, ncols)).T
    if data.ndim == 1:
        return [ [d for d in data] ]
    else:
        return [d for d in data]

def graph_data(data, labels, metric, tsc_filter=None):
    f, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    # Use TSC colour if available, otherwise use a colourmap
    if tsc_filter and tsc_filter in TSC_COLOURS:
        base_colour = TSC_COLOURS[tsc_filter]
    else:
        base_colour = '#333333'

    cmap = plt.colormaps.get_cmap('tab10').resampled(max(len(labels), 2))

    for i, (d, label) in enumerate(zip(data, labels)):
        colour = cmap(i) if len(labels) > 1 else base_colour
        d_arr = np.array(d, dtype=float)
        # Plot raw data faintly
        ax.plot(d_arr, color=colour, alpha=0.15, linewidth=0.5)
        # Plot smoothed line
        ax.plot(smooth(d_arr), color=colour, linewidth=1.8, label=label)

    metric_title = metric.replace('_', ' ').capitalize()
    tsc_name = tsc_filter.upper() if tsc_filter else ''
    ax.set_xlabel('Training Step')
    ax.set_ylabel(metric_title)
    ax.set_title(f'{tsc_name} Training Progress \u2014 {metric_title}')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    save_dir = 'Outputs/plots/'
    os.makedirs(save_dir, exist_ok=True)
    prefix = f'{tsc_filter}_' if tsc_filter else ''
    fname = f'{save_dir}{prefix}{metric}_training.png'
    f.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(f)
    print(f'  Saved: {fname}')

def graph_metric(path, metric, tsc_filter=None):
    files = sorted(os.listdir(path))
    matching = [f for f in files if metric in f]
    if tsc_filter:
        matching = [f for f in matching if f.startswith(tsc_filter + '_')]
    if not matching:
        print(f'No files found for metric={metric}, tsc={tsc_filter}')
        return
    newest_fp = matching[-1]
    print(metric)
    print(newest_fp)
    fp = path + newest_fp
    labels = get_headers(fp)
    data = get_data(fp)
    graph_data(data, labels, metric, tsc_filter)

def parse_cl_args():
    parser = argparse.ArgumentParser(description='Graph RL training progress from tmp/ CSVs')
    parser.add_argument('-tsc', type=str, default=None,
                        help='Filter by TSC type (dqn, ddpg, ppo). Default: newest across all.')
    parser.add_argument('-path', type=str, default='tmp/',
                        help='Path to training CSVs, default: tmp/')
    return parser.parse_args()


def main():
    global_params()
    args = parse_cl_args()

    metrics = ['replay', 'updates', 'nexp']
    for m in metrics:
        graph_metric(args.path, m, args.tsc)


if __name__ == '__main__':
    main()
