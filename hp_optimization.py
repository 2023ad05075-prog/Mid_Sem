"""
Hyperparameter Optimization for Traffic Signal Control.

Grid search over hyperparameter combinations for each TSC type.
Trains (for RL) and tests each combination, ranks by mean+std travel time.

Usage:
    python hp_optimization.py -tsc dqn -sim double -n 4 -l 1
    python hp_optimization.py -tsc ppo -sim double -n 4 -l 1
    python hp_optimization.py -tsc sotl -sim double -n 4
"""

import itertools, time, os, argparse, shutil, subprocess, sys

import numpy as np

from src.picklefuncs import load_data, save_data
from src.helper_funcs import check_and_make_dir, write_lines_to_file, write_line_to_file, get_time_now
from src.controller_types import RL_TSC, SUPPORTED_TSC


def parse_cl_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=4, dest='n',
                        help='number of sim procs, default: 4')
    parser.add_argument("-l", type=int, default=1, dest='l',
                        help='number of learner procs, default: 1')
    parser.add_argument("-sim", type=str, default='double', dest='sim',
                        help='simulation scenario, default: double')
    parser.add_argument("-tsc", type=str, default='dqn', choices=SUPPORTED_TSC, dest='tsc',
                        help='traffic signal control algorithm')
    parser.add_argument("-simlen", type=int, default=3600, dest='sim_len',
                        help='simulation length in seconds, default: 3600')
    parser.add_argument("-port", type=int, default=9000, dest='port',
                        help='base SUMO port, default: 9000')
    parser.add_argument("-updates", type=int, default=5000, dest='updates',
                        help='training updates per HP trial (reduced for speed), default: 5000')

    args = parser.parse_args()
    return args


def get_hp_dict(tsc_str):
    """Define hyperparameter search space for each TSC type."""
    if tsc_str == 'dqn':
        return {
            '-lr': [0.0001, 0.00005],
            '-lre': [0.000001, 0.00000001],
            '-batch': [32, 64],
            '-nreplay': [10000, 15000],
            '-nsteps': [1, 2],
            '-n_hidden': [3],
            '-target_freq': [32, 64, 128],
            '-gmin': [5, 10],
            '-gmax': [30, 45],
        }
    elif tsc_str == 'ddpg':
        return {
            '-lr': [0.0001, 0.00005],
            '-lrc': [0.001, 0.0005],
            '-lre': [0.000001, 0.00000001],
            '-batch': [32, 64],
            '-nreplay': [10000, 15000],
            '-tau': [0.01, 0.005],
            '-n_hidden': [3],
            '-target_freq': [16, 50],
            '-gmax': [30, 45],
        }
    elif tsc_str == 'ppo':
        return {
            '-lr': [0.0001, 0.0003, 0.00005],
            '-lrc': [0.001, 0.0005],
            '-batch': [32, 64],
            '-nreplay': [10000],
            '-n_hidden': [3],
            '-clip': [0.1, 0.2, 0.3],
            '-ppo_epochs': [5, 10],
            '-entropy_coef': [0.01, 0.005],
            '-gmax': [30, 45],
        }
    elif tsc_str == 'sotl':
        return {
            '-theta': [10, 20, 30, 40, 50],
            '-mu': [0, 5, 10, 15],
            '-omega': [0, 5, 10, 15],
        }
    elif tsc_str == 'websters':
        return {
            '-cmin': [40, 60, 80],
            '-cmax': [160, 180, 200],
            '-satflow': [0.3, 0.38, 0.44],
            '-f': [600, 900, 1800],
        }
    else:
        raise ValueError(f'Unknown TSC type: {tsc_str}')


def create_hp_cmds(args, hp_order, hp):
    """Build train and test command strings for a given HP combination."""
    hp_cmds = []

    # Base command with common arguments
    # Note: -gmin and -gmax may be overridden by HP search values appended later
    base_cmd = (
        f'{sys.executable} run_01.py'
        f' -sim {args.sim} -nogui -tsc {args.tsc}'
        f' -simlen {args.sim_len} -port {args.port}'
        f' -scale 1.4 -demand dynamic -offset 0.25'
        f' -y 2 -r 3 -eps 0.01'
        f' -gmin 5 -gmax 30'
        f' -hidden_act elu'
        f' -save_path Outputs/saved_models'
        f' -save_replay Outputs/saved_replays -save_t 120'
    )

    # Append HP values
    hp_str = ' '.join(f'{s} {v}' for s, v in zip(hp_order, hp))

    if args.tsc in RL_TSC:
        # Train command
        train_cmd = (
            f'{base_cmd} {hp_str}'
            f' -mode train -save -n {args.n} -l {args.l}'
            f' -updates {args.updates}'
        )
        hp_cmds.append(train_cmd)

    # Test command — run n+l sims to generate metrics
    test_n = args.n + args.l if args.tsc in RL_TSC else args.n
    test_cmd = f'{base_cmd} {hp_str} -mode test -n {test_n}'
    if args.tsc in RL_TSC:
        test_cmd += ' -load'
    hp_cmds.append(test_cmd)

    return hp_cmds


def get_hp_results(fp):
    """Load travel time results from metrics directory."""
    travel_times = []
    if not os.path.isdir(fp):
        print(f'WARNING: metrics path {fp} not found')
        return travel_times
    for f in os.listdir(fp):
        fpath = os.path.join(fp, f)
        if os.path.isfile(fpath):
            travel_times.extend(load_data(fpath))
    return travel_times


def rank_hp(hp_fitness, hp_order, tsc_str, fp):
    """Rank all HP combinations by mean+std and write results."""
    ranked_hp_fitness = [(hp, hp_fitness[hp]['mean'] + hp_fitness[hp]['std']) for hp in hp_fitness]
    ranked_hp_fitness = sorted(ranked_hp_fitness, key=lambda x: x[-1])

    print(f'\nBest hyperparams set for {tsc_str}')
    print(hp_order)
    print(ranked_hp_fitness[0])

    lines = [','.join(hp_order) + ',mean,std,mean+std']
    for hp_val, score in ranked_hp_fitness:
        lines.append(f"{hp_val},{hp_fitness[hp_val]['mean']},{hp_fitness[hp_val]['std']},{score}")

    write_lines_to_file(fp, 'a+', lines)


def write_temp_hp(hp, results, fp):
    write_line_to_file(fp, 'a+', f"{hp},{results['mean']},{results['std']},{results['mean']+results['std']}")


def save_hp_performance(data, path, hp_str):
    check_and_make_dir(path)
    save_data(os.path.join(path, hp_str + '.p'), data)


def run_command(cmd):
    """Run a command and return the exit code."""
    print(f'\n>>> {cmd}\n')
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    start = time.time()

    args = parse_cl_args()
    tsc_str = args.tsc

    hp_dict = get_hp_dict(tsc_str)
    hp_order = sorted(list(hp_dict.keys()))

    hp_list = [hp_dict[hp] for hp in hp_order]
    hp_set = list(itertools.product(*hp_list))
    print(f'{len(hp_set)} total hyperparameter combinations for {tsc_str}')
    print(f'Parameters: {hp_order}')

    hp_travel_times = {}
    # Metrics path includes mode subfolder (train/test separation)
    metrics_fp = f'Outputs/metrics/{tsc_str}/test'

    # Output results directory
    path = f'hyperparams/{tsc_str}/'
    check_and_make_dir(path)
    fname = get_time_now()
    hp_fp = path + fname + '.csv'
    write_line_to_file(hp_fp, 'a+', ','.join(hp_order) + ',mean,std,mean+std')

    for i, hp in enumerate(hp_set, 1):
        print(f'\n{"="*60}')
        print(f'HP TRIAL {i}/{len(hp_set)}')
        print(f'{"="*60}')
        for param, val in zip(hp_order, hp):
            print(f'  {param}: {val}')

        hp_cmds = create_hp_cmds(args, hp_order, hp)

        success = True
        for cmd in hp_cmds:
            ret = run_command(cmd)
            if ret != 0:
                print(f'WARNING: command failed with exit code {ret}')
                success = False
                break

        hp_str = ','.join([str(h) for h in hp])

        if success:
            travel_time_path = os.path.join(metrics_fp, 'traveltime')
            travel_times = get_hp_results(travel_time_path)

            if len(travel_times) > 0:
                hp_travel_times[hp_str] = {
                    'mean': int(np.mean(travel_times)),
                    'std': int(np.std(travel_times))
                }
                write_temp_hp(hp_str, hp_travel_times[hp_str], hp_fp)
                save_hp_performance(travel_times, f'hp/{tsc_str}/', hp_str)
            else:
                print(f'WARNING: no travel time data found for HP: {hp_str}')
                hp_travel_times[hp_str] = {'mean': 99999, 'std': 99999}
        else:
            hp_travel_times[hp_str] = {'mean': 99999, 'std': 99999}

        # Clean metrics for this trial to avoid contaminating next
        if os.path.isdir(metrics_fp):
            shutil.rmtree(metrics_fp)

        # Clean saved models between trials so next trial trains fresh
        model_path = f'Outputs/saved_models/{tsc_str}'
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)

    # Write final ranked results
    if os.path.isfile(hp_fp):
        os.remove(hp_fp)
    rank_hp(hp_travel_times, hp_order, tsc_str, hp_fp)
    print(f'\nAll hyperparameter results saved to: {hp_fp}')

    secs = time.time() - start
    print(f'\nTOTAL HP SEARCH TIME: {int(secs/60)} minutes')


if __name__ == '__main__':
    main()
