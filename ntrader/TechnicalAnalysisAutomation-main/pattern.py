Certainly! I've rewritten the code to be more suitable for a Jupyter notebook environment. I've removed the class structure and organized the functionality into separate functions. Additionally, I've added comments to explain each part of the code. Please note that you'll need to run the previous imports in your Jupyter notebook before executing this code.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from perceptually_important import find_pips

def load_and_preprocess_data(file_path):
    # Load data from CSV file
    data = pd.read_csv(file_path)
    
    # Convert 'date' column to datetime and set it as the index
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    
    # Take the logarithm of the closing prices
    data = np.log(data)
    
    return data

def plot_cluster_examples(candle_data, unique_pip_indices, pip_clusters, lookback, n_pips=5, grid_size=5):
    plt.style.use('dark_background')
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    flat_axs = axs.flatten()

    for i, ax in enumerate(flat_axs):
        if i >= len(unique_pip_indices):
            break
            
        pat_i = unique_pip_indices[pip_clusters[i]]
        data_slice = candle_data.iloc[pat_i - lookback + 1: pat_i + 1]
        idx = data_slice.index
        plot_pip_x, plot_pip_y = find_pips(data_slice['close'].to_numpy(), n_pips, 3)
        
        pip_lines = []
        colors = []
        for line_i in range(n_pips - 1):
            l0 = [(idx[plot_pip_x[line_i]], plot_pip_y[line_i]),
                  (idx[plot_pip_x[line_i + 1]], plot_pip_y[line_i + 1])]
            pip_lines.append(l0)
            colors.append('w')

        mpf.plot(data_slice, type='candle',
                 alines=dict(alines=pip_lines, colors=colors),
                 ax=ax, style='charles', update_width_config=dict(candle_linewidth=1.75))
        
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")

    fig.suptitle("Cluster Examples", fontsize=32)
    plt.show()

def train_and_evaluate_model(data, n_pips=5, lookback=24, hold_period=6, n_reps=-1):
    unique_pip_patterns, unique_pip_indices = find_unique_patterns(data['close'].to_numpy(), n_pips, lookback)
    amount_clusters = find_optimal_clusters(unique_pip_patterns)
    pip_clusters, cluster_centers = kmeans_cluster_patterns(unique_pip_patterns, amount_clusters)
    cluster_signals = get_cluster_signals(pip_clusters, unique_pip_indices, hold_period, len(data))
    selected_long, selected_short = assign_clusters(cluster_signals, n_selected=1)
    fit_martin = get_total_performance(data, unique_pip_indices, pip_clusters, selected_long, selected_short)

    print("Fit Martin:", fit_martin)

    if n_reps > 1:
        perm_martins = monte_carlo_permutation_test(data, unique_pip_patterns, unique_pip_indices,
                                                    n_pips, lookback, hold_period, n_reps)
        print("Permutation Martins:", perm_martins)

def find_unique_patterns(data_array, n_pips, lookback):
    unique_pip_indices = []
    unique_pip_patterns = []
    
    last_pips_x = [0] * n_pips
    
    for i in range(lookback - 1, len(data_array)):
        start_i = i - lookback + 1
        window = data_array[start_i: i + 1]
        pips_x, pips_y = find_pips(window, n_pips, 3)
        pips_x = [j + start_i for j in pips_x]

        same = all(pips_x[j] == last_pips_x[j] for j in range(1, n_pips - 1))

        if not same:
            pips_y = list((np.array(pips_y) - np.mean(pips_y)) / np.std(pips_y))
            unique_pip_patterns.append(pips_y)
            unique_pip_indices.append(i)

        last_pips_x = pips_x

    return unique_pip_patterns, unique_pip_indices

def find_optimal_clusters(unique_pip_patterns):
    search_instance = silhouette_ksearch(
        unique_pip_patterns, 5, 40, algorithm=silhouette_ksearch_type.KMEANS
    ).process()
    return search_instance.get_amount()

def kmeans_cluster_patterns(unique_pip_patterns, amount_clusters):
    initial_centers = kmeans_plusplus_initializer(unique_pip_patterns, amount_clusters).initialize()
    kmeans_instance = kmeans(unique_pip_patterns, initial_centers)
    kmeans_instance.process()
    
    return kmeans_instance.get_clusters(), kmeans_instance.get_centers()

def get_cluster_signals(pip_clusters, unique_pip_indices, hold_period, data_length):
    cluster_signals = []

    for clust in pip_clusters:
        signal = np.zeros(data_length)
        for mem in clust:
            arr_i = unique_pip_indices[mem]
            signal[arr_i: arr_i + hold_period] = 1. 

        cluster_signals.append(signal)

    return cluster_signals

def assign_clusters(cluster_signals, n_selected=1):
    selected_long = np.argsort([np.sum(signal) for signal in cluster_signals])[-n_selected:]
    selected_short = np.argsort([np.sum(signal) for signal in cluster_signals])[:n_selected]

    return selected_long, selected_short

def get_total_performance(data, unique_pip_indices, pip_clusters, selected_long, selected_short):
    long_signal = np.zeros(len(data))
    short_signal = np.zeros(len(data))

    for clust_i in range(len(pip_clusters)):
        if clust_i in selected_long:
            long_signal += pip_clusters[clust_i]
        elif clust_i in selected_short:
            short_signal += pip_clusters[clust_i]

    long_signal /= len(selected_long)
    short_signal /= len(selected_short)
    short_signal *= -1

    rets = (long_signal + short_signal) * data['close'].diff().shift(-1)

    rsum = np.sum(rets)
    martin = rsum / np.sqrt(np.sum((np.cumsum(rets) / np.cummax(np.exp(np.cumsum(rets)))) ** 2))

    return martin

def monte_carlo_permutation_test(data, unique_pip_patterns, unique_pip_indices,
                                  n_pips, lookback, hold_period, n_reps):
    perm_martins = []

    for rep in range(1, n_reps):
        data_copy = data.copy()
        returns_copy

 = data['close'].diff().shift(-1).copy()

        np.random.shuffle(returns_copy)
        data_copy['close'] = np.cumsum(np.concatenate([np.array([data_copy['close'].iloc[0]]), returns_copy]))

        unique_pip_patterns, unique_pip_indices = find_unique_patterns(data_copy['close'].to_numpy(), n_pips, lookback)
        amount_clusters = find_optimal_clusters(unique_pip_patterns)
        pip_clusters, cluster_centers = kmeans_cluster_patterns(unique_pip_patterns, amount_clusters)
        cluster_signals = get_cluster_signals(pip_clusters, unique_pip_indices, hold_period, len(data_copy))
        selected_long, selected_short = assign_clusters(cluster_signals, n_selected=1)
        perm_martin = get_total_performance(data_copy, unique_pip_indices, pip_clusters, selected_long, selected_short)
        perm_martins.append(perm_martin)

    return perm_martins

# Load and preprocess data
file_path = 'BTCUSDT3600.csv'
data = load_and_preprocess_data(file_path)

# Visualize cluster examples
unique_pip_patterns, unique_pip_indices = find_unique_patterns(data['close'].to_numpy(), n_pips=5, lookback=24)
amount_clusters = find_optimal_clusters(unique_pip_patterns)
pip_clusters, cluster_centers = kmeans_cluster_patterns(unique_pip_patterns, amount_clusters)
plot_cluster_examples(data, unique_pip_indices, pip_clusters, lookback=24, n_pips=5, grid_size=5)

# Train and evaluate the model
train_and_evaluate_model(data, n_pips=5, lookback=24, hold_period=6, n_reps=5)
```

This code organizes the functionality into functions and separates the training and evaluation steps. You can run each function separately and observe the results.