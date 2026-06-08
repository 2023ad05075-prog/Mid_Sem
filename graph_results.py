import os, argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mp
from matplotlib.colors import LinearSegmentedColormap

from src.graph_globals import global_params, TSC_COLOURS, TSC_LABELS
from src.graphs import graph, boxplot, multi_line, multi_line_with_CI, get_cmap, scatter, save_graph
from src.picklefuncs import load_data
from src.helper_funcs import check_and_make_dir

def main():
    global_params()
    labels = TSC_LABELS
    colours = TSC_COLOURS

    args = parse_cl_args()
    check_and_make_dir(args.save_dir)

    if args.type == 'moe':
        fp = 'Outputs/metrics/'
        graph_travel_time(labels, colours, fp, args.save_dir, args.mode)
        metrics = ['queue', 'delay']
        graph_individual_intersections(labels, colours, fp, metrics, args.save_dir, args.mode)
    elif args.type == 'hp': 
        fp = 'hp/'
        graph_hyper_params(labels, colours, fp, args.save_dir)
    else:
        assert 0, print('Error, supplied graph type argument '+str(args.type)+' does not exist')

def parse_cl_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-type", type=str, default='moe', dest='type', help='Data to be graphed, default: moe, options: moe, hp')
    parser.add_argument("-save_dir", type=str, default='Outputs/figures/', dest='save_dir', help='Directory to save figures, default: Outputs/figures/')
    parser.add_argument("-mode", type=str, default='test', dest='mode', help='Metrics mode subfolder (train/test). Default: test')
    args = parser.parse_args()
    return args


def resolve_metric_path(base_fp, tsc, mode, *parts):
    """Resolve metric path: try <tsc>/<mode>/<parts> first, fall back to <tsc>/<parts> (legacy)."""
    with_mode = os.path.join(base_fp, tsc, mode, *parts)
    if os.path.exists(with_mode):
        return with_mode
    without_mode = os.path.join(base_fp, tsc, *parts)
    if os.path.exists(without_mode):
        return without_mode
    return with_mode  # return mode path for error messages

def graph_hyper_params(labels, colours, fp, save_dir):
    tsc = os.listdir(fp)
    tsc_hp = {}
                                                                                                                   
    #get data
    for t in tsc:
        tsc_fp = fp+t+'/'
        data = [ load_data(tsc_fp+f) for f in os.listdir(tsc_fp)]                                                 
        tsc_hp[t] = np.stack([ [np.mean(d), np.std(d)] for d in data]).T

    #create appropriate graph
    n = len(tsc)
    if n == 1:
        f, axes = plt.subplots()
        axes = [axes]
    else:
        nrows = 2 
        ncols = int(n/nrows) if n%nrows == 0 else int((n+1)/nrows)
        f, axes = plt.subplots(nrows=nrows,ncols=ncols)
        axes = axes.flat
    
    if n%nrows != 0:
        f.delaxes(axes[-1])

    XTITLE = 'Mean Travel Time (s)'
    YTITLE = 'Std. Dev. Travel Time (s)'

    #graph each tsc hyperparemeter
    for ax, t, i in zip(axes, tsc, range(len(tsc))):
        #order hp performance from low to high 
        #w.r.t mean+std
        mean_data = tsc_hp[t][0] 
        std_data = tsc_hp[t][1]
        data = sorted([ (m+s, m, s) for m,s in zip(mean_data, std_data) ], key = lambda x:x[0] )
        data = np.stack([ [d[1], d[2]] for d in data]).T
        mean_data = data[0]
        std_data = data[1]

        #rainbow_colours = mp.cm.rainbow(np.linspace(0, 1, len(mean_data)))                          

        rg_colours = mp.cm.brg(np.linspace(1.0, 0.5, len(mean_data)))                          
        if i%ncols == 0 and i >= len(tsc)/2:
            xtitle = XTITLE 
            ytitle = YTITLE 
        elif i%ncols == 0:
            xtitle = ''
            ytitle = YTITLE
        elif i >= len(tsc)/2:
            xtitle = XTITLE
            ytitle = ''
        else:
            xtitle = ''
            ytitle = ''
        #graph each tsc hp performance 
        graph( ax, mean_data, scatter( ax, mean_data, std_data, rg_colours, ['']*len(mean_data)), 
                                  xtitle=xtitle,                                 
                                  ytitle_pad = ytitle,       
                                  title=str(labels[t]),        
                                  xlim = [0.0, max(mean_data)*1.05],                          
                                  ylim= [0.0, max(std_data)*1.05],                            
                                  grid=True)                                                  

    #axis colourbar
    cax = f.add_axes([0.915, 0.1, 0.05, 0.85])
    cmap = mp.cm.brg
    cm = LinearSegmentedColormap.from_list('rg', rg_colours, N=rg_colours.shape[0])
    norm = mp.colors.Normalize(vmin=0.5, vmax=1.0)
    cb = mp.colorbar.ColorbarBase(cax, cmap=cm,
                                norm=norm,
                                orientation='vertical')

    #color bar axis text
    #print([ l._text for l in cb.ax.get_yticklabels()])
    #cb_labels = ['']*rg_colours.shape[0]
    cb_labels = [ l._text for l in cb.ax.get_yticklabels()]
    cb_labels[0] = 'Best'
    cb_labels[-1] = 'Worst'
    cb.ax.set_yticklabels(cb_labels)

    f.suptitle('Hyperparameter Performance', fontsize=14, fontweight='bold')                                                            
    save_graph(f, save_dir+'tsc_hp.pdf')
    plt.close(f)

    #now compare all tsc hp sets together in one graph
    #prepare data
    data_order = sorted(tsc_hp.keys())
    #tsc_color = colours[:len(data_order)]
    mean_data, std_data, colors, tsc_labels = [], [], [], []
    for d in data_order:
        n = len(tsc_hp[d][0])
        mean_data.extend(tsc_hp[d][0])
        std_data.extend(tsc_hp[d][1])
        tsc_labels.extend(labels[d])
        #colors.extend([c]*len(tsc_hp[d][0]))
        colors.extend( [colours[d]]*n )
    #graph all hp data all together
    f, ax = plt.subplots(1,1)
    graph( ax, mean_data, scatter( ax, mean_data, std_data, colors, ['']*len(mean_data)),
                              xtitle=XTITLE,
                              ytitle_pad = YTITLE,
                              title='Traffic Signal Control\nHyperparameter Comparison',
                              xlim = [0.0, 200.0],
                              ylim= [0.0, 200.0],
                              #xlim = [0.0, max(mean_data)*1.05],
                              #ylim= [0.0, max(std_data)*1.05],
                              #legend=(0.82, 0.72),                                   
                              #colours=colours,
                              grid=True)

    #colorbar

    #add legend manually because we only
    #want one for each tsc
    patches = []
    for d in data_order:
        c = colours[d]
        patches.append( mpatches.Patch(color=c, label=labels[d]) )
    plt.legend(handles=patches, framealpha=0.9)
    save_graph(f, save_dir+'hp.pdf')
    plt.close(f)

def graph_travel_time(labels, colours, fp, save_dir, mode='test'):
    #read metric data for all tsc types                                           
    data = get_data(fp, 'traveltime', get_folder_data, mode)                               
    #prepare data for graph                                                          
    data_order = sorted(data.keys())                                                 
    data = [ data[d] for d in data_order]                                           
    plot_labels = [ labels[d] for d in data_order]                                       
    c = [ colours[d] for d in data_order]                                       
    #graph data                                                                      
    f, ax = plt.subplots(1, 1, figsize=(8, 5))                                                        
    graph( ax, data, boxplot( ax, data, c, plot_labels),                            
                              xtitle='Traffic Signal Controller',                    
                              ytitle_pad='Travel Time (s)',                      
                              title='Travel Time Distribution by Controller',     
                              grid=True)                                             

    # Annotate with mean, std, median below each box
    for i, d in enumerate(data_order):
        mu = int(np.mean(data[i]))
        sigma = int(np.std(data[i]))
        med = int(np.median(data[i]))
        text = f'$\\mu$={mu}\n$\\sigma$={sigma}\nmed={med}'
        ymin = ax.get_ylim()[0]
        ax.text(i + 1, ax.get_ylim()[1] * 0.92, text,
                ha='center', va='top', fontsize=8,
                color=c[i], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=c[i], alpha=0.8))

    save_graph(f, save_dir+'travel_time.pdf')
    plt.close(f)
    print(f'Saved: {save_dir}travel_time.pdf')

def graph_conf_interval(labels, colours, fp, metric, mode='test'):
    #read metric data for all tsc types                               
    data = get_data(fp, metric, get_metric_data, mode)                         
    #prepare data for graph                                              
    data_order = sorted(data.keys())                                     
    data = [ data[d]  for d in data_order]                               
    plot_labels = [ labels[d]  for d in data_order]                           
    #graph data                                                          
    f, ax = plt.subplots(1, 1, figsize=(8, 5))                                            
    metric_title = metric.capitalize()                                   
    graph( ax, data, multi_line_with_CI( ax, data, colours, plot_labels),     
           xtitle='Time (s)',                                            
           ytitle_pad=metric_title,                              
           title=f'{metric_title} by Traffic Signal Controller',          
           legend='upper left',                                          
           grid=True)                                                    
    save_graph(f, f'{fp}{metric}_ci.pdf')
    plt.close(f)

def get_data(fp, metric, read_data_func, mode='test'):
    tsc = os.listdir(fp)
    tsc_data = {}
    for t in tsc:
        metric_path = resolve_metric_path(fp, t, mode, metric)
        if os.path.exists(metric_path):
            tsc_data[t] = read_data_func(metric_path)
        else:
            print(f'WARNING: no {metric} data for {t} at {metric_path}, skipping')
    return tsc_data

def get_metric_data(fp):
    #for use with queue and delay data
    #sort all metric data from same tsc_id 
    if not os.path.exists(fp):
        assert 0, 'Supplied path '+str(fp)+' does not exist.'

    tsc_data = {tsc_id:sorted(os.listdir(fp+'/'+tsc_id)) 
                    for tsc_id in os.listdir(fp)}
    sim_runs_data = []

    #until all data has been popped
    k = list(tsc_data.keys())[0]
    while len(tsc_data[k]) > 0:
        #get file path for each intersection from same sim run
        same_run_data = [ fp+'/'+tsc_id+'/'+tsc_data[tsc_id].pop(0) 
                          for tsc_id in tsc_data ]
        same_run_data = [ load_data(f) for f in same_run_data ]
        #sum across time axis, each element of array
        #represents the sum of all tsc_id metric
        sim_runs_data.append( np.sum(same_run_data, axis=0) )

    return np.stack(sim_runs_data)

def get_folder_data(fp):
    #all the travel times can be
    #grouped together by extending list
    if not os.path.exists(fp):
        assert 0, 'Supplied path '+str(fp)+' does not exist.'
    data = []
    for f in os.listdir(fp):
        data.extend(load_data(fp+'/'+f))
    return np.array(data)

def stack_folder_files(fp):
    data = [ load_data(fp+f) for f in os.listdir(fp)]
    return np.stack(data)

def graph_individual_intersections(labels, colours, fp, metrics, save_dir, mode='test'):
    #rows are metrics
    #columns are intersections

    # Only include TSCs that have the required metric data and are in labels
    all_tsc = os.listdir(fp)
    tsc = []
    for t in all_tsc:
        metric_path = resolve_metric_path(fp, t, mode, metrics[0])
        if os.path.exists(metric_path) and t in labels:
            tsc.append(t)
    if not tsc:
        print('No TSC data found for intersection metrics')
        return

    # Get intersections from first available TSC
    first_path = resolve_metric_path(fp, tsc[0], mode, metrics[0])
    intersections = os.listdir(first_path)
    ncols = len(intersections)
    nrows = len(metrics)

    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5*nrows))
    if ax.ndim == 1:
        ax = ax[...,np.newaxis]

    for m, r in zip(metrics, range(nrows)):
        #get metric data for each intersection
        data = {}
        for t in tsc:
            data[t] = {}
            for i in intersections:
                alias_p = 60
                data_path = resolve_metric_path(fp, t, mode, m, i)
                data[t][i] = alias( stack_folder_files(data_path+'/'), alias_p)

        xtitle = 'Time (min)' if r == nrows-1 else ''
        #graph same metric for each intersection
        for I, c in zip(intersections, range(ncols)):
            title = I if r == 0 else ''
            if m == 'queue':
                ytitle = 'Queue (veh)' if c == 0 else ''
            else:
                ytitle = 'Delay (s)' if c == 0 else ''
            legend = 'upper left'
                
            data_order = sorted(data.keys())               
            alias_p = 60 
            order_data = [ data[d][I] for d in data_order]     

            order_labels = [ labels[d] for d in data_order]
            order_colors = [ colours[d] for d in data_order]

            label_c = { labels[d]:colours[d] for d in data_order}
            graph( ax[r,c], order_data, multi_line_with_CI( ax[r,c], order_data, order_colors, order_labels),
                   xtitle=xtitle,
                   ytitle_pad=ytitle,
                   title=title,
                   legend=legend,
                   colours = label_c,
                   grid=True)
            ax[r,c].set_xlim(left=0)
            ax[r,c].set_ylim(bottom=0)
    f.suptitle('Intersection Measures of Effectiveness', fontsize=14, fontweight='bold')
    f.tight_layout(rect=[0, 0, 1, 0.95])
    save_graph(f, save_dir+'intersection_moe.pdf')
    plt.close(f)
    print(f'Saved: {save_dir}intersection_moe.pdf')

def alias(data, a):
    n = data.shape[-1]
    if n % a != 0:
        #error?
        a = 1

    stop = n-a
    m = int(n/a)

    alias_data = []
    for d in data:
        #alias data timeseries
        alias_data.append( np.array([np.sum(d[i*a:(i+1)*a]) for i in range(m) ]) )

    return np.stack(alias_data)
    #return np.stack([np.sum(data[i*a:(i+1)*a]) for i in range(stop) ])
    #return np.array([ np.sum(data[i*a:(i+1)*a]) for i in range(stop)])

if __name__ == '__main__':
    main()
