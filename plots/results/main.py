import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json, glob, re
import numpy as np
import pandas as pd

from classes import GenerationMetrics, GenerationOutput, GenerationResult
from collections import Counter, defaultdict

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()
sns.color_palette("colorblind")

RESULTS_DIR = '/Users/thijs/Documents/Git/msc-experiment/results'

LABELS = ['Function-level, Small Model', 'Line-level, Small Model', 'Function-level, Large Model', 'Line-level, Large Model']

def load_data(file_name):
    output = defaultdict(list)

    for file in glob.glob(f'{RESULTS_DIR}/**/{file_name}.jsonl'):
        problem_id = re.search(r'(\d+)', file)[0]

        with open(file) as f:
            for line in f.read().splitlines():
                result = GenerationResult(**json.loads(line))
                result.outputs = [GenerationOutput(**o) for o in result.outputs]
                result.metrics = GenerationMetrics(**result.metrics)

                output[problem_id].append(result)

    return output

def reject_outliers(x, m=2.):
    d = np.abs(x - np.median(x))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(x))
    return x[s < m]

def reject_outliers_multiple(x, y, m=2.):
    x_d = np.abs(x - np.median(x))
    x_mdev = np.median(x_d)
    x_s = x_d / x_mdev if x_mdev else np.zeros(len(x))
    
    y_d = np.abs(y - np.median(y))
    y_mdev = np.median(y_d)
    y_s = y_d / y_mdev if y_mdev else np.zeros(len(y))

    valid_mask = (x_s < m) & (y_s < m)

    return x[valid_mask], y[valid_mask]

def energy_violinplot(func_small, line_small, func_large, line_large):
    fig, ax = plt.subplots(figsize=(8,4))

    palette = sns.color_palette()

    def plot(data, get_datapoint, index, label, color):
        xs = []

        for problem_id, results in data.items():
            for result in results:
                xs.append(get_datapoint(result))

        avg = sum(xs) / len(xs)

        filtered_xs = reject_outliers(np.array(xs), m=10.)

        df = pd.DataFrame({'Value': filtered_xs, 'Category': [label] * len(filtered_xs)})
        sns.violinplot(x='Category', y='Value', data=df, ax=ax, inner=None, color=color)

    plot(func_small, get_datapoint=lambda x: x.metrics.energy, index=0, label=LABELS[0], color=palette[0])
    plot(line_small, get_datapoint=lambda x: x.metrics.energy, index=1, label=LABELS[1], color=palette[1])
    plot(func_large, get_datapoint=lambda x: x.metrics.energy, index=2, label=LABELS[2], color=palette[2])
    plot(line_large, get_datapoint=lambda x: x.metrics.energy, index=3, label=LABELS[3], color=palette[3])

    plt.legend(labels=LABELS, loc='upper left')
    plt.xticks([])
    plt.title('Distribution of Energy (J)')
    plt.xlabel(None)
    plt.ylabel('Energy (J)')

    plt.tight_layout()
    plt.savefig('output/energy-violinplot')

    plt.cla()

    plot(func_small, get_datapoint=lambda x: x.metrics.energy_per_token, index=0, label=LABELS[0], color=palette[0])
    plot(line_small, get_datapoint=lambda x: x.metrics.energy_per_token, index=1, label=LABELS[1], color=palette[1])
    plot(func_large, get_datapoint=lambda x: x.metrics.energy_per_token, index=2, label=LABELS[2], color=palette[2])
    plot(line_large, get_datapoint=lambda x: x.metrics.energy_per_token, index=3, label=LABELS[3], color=palette[3])

    plt.xticks([])
    plt.title('Distribution of Energy per Token (J)')
    plt.xlabel(None)
    plt.ylabel('Energy per Token (J)')
    plt.legend(labels=LABELS, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/energy-per-token-violinplot')

def time_violinplot(func_small, line_small, func_large, line_large):
    fig, ax = plt.subplots(figsize=(8,4))
    palette = sns.color_palette()

    def plot(data, get_datapoint, index, label, color):
        xs = []

        for problem_id, results in data.items():
            for result in results:
                xs.append(get_datapoint(result))

        avg = sum(xs) / len(xs)

        filtered_xs = reject_outliers(np.array(xs), m=10.)

        df = pd.DataFrame({'Value': filtered_xs, 'Category': [label] * len(filtered_xs)})
        sns.violinplot(x='Category', y='Value', data=df, ax=ax, inner=None, color=color)

    plot(func_small, get_datapoint=lambda x: x.metrics.time, index=0, label=LABELS[0], color=palette[0])
    plot(line_small, get_datapoint=lambda x: x.metrics.time, index=1, label=LABELS[1], color=palette[1])
    plot(func_large, get_datapoint=lambda x: x.metrics.time, index=2, label=LABELS[2], color=palette[2])
    plot(line_large, get_datapoint=lambda x: x.metrics.time, index=3, label=LABELS[3], color=palette[3])

    plt.legend(labels=LABELS, loc='upper left')
    plt.xticks([])
    plt.title('Distribution of Time (s)')
    plt.xlabel(None)
    plt.ylabel('Time (s)')

    plt.tight_layout()
    plt.savefig('output/time-violinplot')

    plt.cla()

    plot(func_small, get_datapoint=lambda x: x.metrics.time_per_token, index=0, label=LABELS[0], color=palette[0])
    plot(line_small, get_datapoint=lambda x: x.metrics.time_per_token, index=1, label=LABELS[1], color=palette[1])
    plot(func_large, get_datapoint=lambda x: x.metrics.time_per_token, index=2, label=LABELS[2], color=palette[2])
    plot(line_large, get_datapoint=lambda x: x.metrics.time_per_token, index=3, label=LABELS[3], color=palette[3])

    plt.legend(labels=LABELS, loc='upper left')
    plt.xticks([])
    plt.title('Distribution of Time per Token (s)')
    plt.xlabel(None)
    plt.ylabel('Time per Token (s)')
    
    plt.tight_layout()
    plt.savefig('output/time-per-token-violinplot')

def finish_reason_barplot(func_small, func_large):
    def get_finish_reasons(data):
        stop = 0
        length = 0

        for problem_id, results in data.items():
            for result in results:
                for output in result.outputs:
                    if output.finish_reason == 'stop':
                        stop += 1
                    else:
                        length += 1

        return stop / (stop + length) * 100, 100

    small_stop, small_length = get_finish_reasons(func_small)
    large_stop, large_length = get_finish_reasons(func_large)

    df = pd.DataFrame({
        'Category': ['Small Model', 'Small Model', 'Large Model', 'Large Model'],
        'Finish Reason': ['Stop', 'Length', 'Stop', 'Length'],
        'Count': [small_stop, small_length, large_stop, large_length]
    })

    df_pivot = df.pivot_table(index='Category', columns='Finish Reason', values='Count', aggfunc='sum', fill_value=0)

    plt.figure(figsize=(6,4))

    sns.barplot(x=df_pivot.index, y=df_pivot['Length'], label='Length', alpha=1, order=['Small Model', 'Large Model'])
    sns.barplot(x=df_pivot.index, y=df_pivot['Stop'], label='Stop', alpha=1, order=['Small Model', 'Large Model'])

    plt.xlabel(None)
    plt.ylabel('Percentage (%)')
    plt.title('Finish Reasons of Function-Level Generations')
    
    plt.tight_layout()
    plt.savefig('output/finish-reasons-barplot')

def energy_vs_time_scatterplot(func_small, line_small, func_large, line_large):
    fig, ax = plt.subplots(figsize=(8,4))

    def plot(data, label, get_energy, get_time):
        xs = []
        ys = []

        for problem_id, results in data.items():
            xx = []
            yy = []

            for result in results:
                xx.append(get_energy(result))
                yy.append(get_time(result))

            xx, yy = reject_outliers_multiple(np.array(xx), np.array(yy))

            x = sum(xx) / len(xx)
            y = sum(yy) / len(yy)

            xs.append(x)
            ys.append(y)

        sns.scatterplot(x=xs, y=ys, s=10, label=label, alpha=0.7, ax=ax)

    # TIME PER TOKEN VS ENERGY PER TOKEN

    get_energy = lambda x: x.metrics.energy_per_token
    get_time = lambda x: x.metrics.time_per_token

    plot(func_small, LABELS[0], get_energy, get_time)
    plot(line_small, LABELS[1], get_energy, get_time)
    plot(func_large, LABELS[2], get_energy, get_time)
    plot(line_large, LABELS[3], get_energy, get_time)

    plt.title('Energy per Token (J) vs. Time per Token (s)')

    plt.xlim(0, 10)
    plt.ylim(0, 0.33)

    plt.xlabel('Energy per Token (J)')
    plt.ylabel('Time per Token (s)')

    plt.legend(labels=LABELS)
    plt.tight_layout()
    plt.savefig('output/energy-per-token-vs-time-per-token-scatterplot')
    plt.cla()

    # TIME VS ENERGY

    get_energy = lambda x: x.metrics.energy
    get_time = lambda x: x.metrics.time

    plot(func_small, LABELS[0], get_energy, get_time)
    plot(line_small, LABELS[1], get_energy, get_time)
    plot(func_large, LABELS[2], get_energy, get_time)
    plot(line_large, LABELS[3], get_energy, get_time)

    plt.title('Energy (J) vs. Time (s)')

    plt.xlabel('Energy (J)')
    plt.ylabel('Time (s)')

    plt.legend(labels=LABELS)
    plt.tight_layout()
    plt.savefig('output/energy-vs-time-scatterplot')

def distribution_unique_solutions(line_small, line_large):
    fig, ax = plt.subplots(figsize=(8,4))

    def prepare_data(data):
        unique_solutions_counter = Counter()

        for problem_id, results in data.items():
            line_counts = defaultdict(set)
            for result in results:
                for i, output in enumerate(result.outputs):
                    line_counts[i].add(output.text)
            for _, counts in line_counts.items():
                unique_solutions_counter[len(counts)] += 1

        solution_keys = list(unique_solutions_counter.keys())
        solution_keys.sort()

        count_data = [unique_solutions_counter.get(key, 0) for key in solution_keys]
        
        return pd.Series(count_data, index=solution_keys)

    df = pd.DataFrame({ 'Small Model (1.5B)': prepare_data(line_small), 'Large Model (9B)': prepare_data(line_large) })
    df_reset = df.reset_index()
    df_reset.columns = ['Number of Unique Solutions', 'Small Model (1.5B)', 'Large Model (9B)']
    df_melted = df_reset.melt(id_vars=['Number of Unique Solutions'], value_vars=['Small Model (1.5B)', 'Large Model (9B)'], var_name='Dataset', value_name='Counts')

    sns.histplot(data=df_melted, x='Number of Unique Solutions', hue='Dataset', weights='Counts', multiple="dodge", bins=30, ax=ax)

    plt.xlim(xmin=1)
    plt.title('Total Unique Solutions per Line')
    plt.xlabel('Number of Unique Solutions')
    plt.tight_layout()
    plt.savefig('output/distribution-unique-line-generations')

def cumulative_distribution_function_time(func_small, line_small, func_large, line_large):
    fig, ax = plt.subplots(figsize=(8,4))

    def normalize(data):
        min_val = min(data)
        max_val = max(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

    def plot(data, label):
        average_times = []

        for problem_id, results in data.items():
            times = []
            for result in results:
                times.append(result.metrics.time)
            average_times.append(sum(times) / len(times))

        sns.ecdfplot(normalize(average_times), ax=ax, label=label)

    plot(func_small, label=LABELS[0])
    plot(line_small, label=LABELS[1])
    plot(func_large, label=LABELS[2])
    plot(line_large, label=LABELS[3])

    plt.xlabel('Normalised Time')
    plt.xlim(0, 1)
    plt.ylim(-0.025, 1.025)
    plt.legend(labels=LABELS)
    plt.title('Cumulative Distribution Function of Time')
    plt.tight_layout()
    plt.savefig('output/cumulative-distribution-function-time')

def cumulative_distribution_function_energy(func_small, line_small, func_large, line_large):
    fig, ax = plt.subplots(figsize=(8,4))

    def normalize(data):
        min_val = min(data)
        max_val = max(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

    def plot(data, label):
        average_energies = []

        for problem_id, results in data.items():
            energies = []
            for result in results:
                energies.append(result.metrics.energy)
            average_energies.append(sum(energies) / len(energies))

        sns.ecdfplot(normalize(average_energies), ax=ax, label=label)

    plot(func_small, label=LABELS[0])
    plot(line_small, label=LABELS[1])
    plot(func_large, label=LABELS[2])
    plot(line_large, label=LABELS[3])

    plt.xlabel('Normalised Energy')
    plt.xlim(0, 1)
    plt.ylim(-0.025, 1.025)
    plt.legend(labels=LABELS)
    plt.title('Cumulative Distribution Function of Energy')
    plt.tight_layout()
    plt.savefig('output/cumulative-distribution-function-energy')

def logprobs_density_plot(func_small, line_small, func_large, line_large):
    fig, ax = plt.subplots(figsize=(8,4))

    def plot(data, label):
        logprobs = []

        for results in data.values():
            for result in results:
                for output in result.outputs:
                    logprobs.extend(output.logprobs)

        probabilities = np.power(10, logprobs)

        sns.kdeplot(probabilities, ax=ax, label=label)

    plot(func_small, label=LABELS[0])
    plot(line_small, label=LABELS[1])

    plt.title('Probability Density Distribution (Small Model)')
    plt.xlabel('Probability')
    plt.legend(labels=[LABELS[0], LABELS[1]])

    plt.tight_layout()
    plt.savefig('output/logprobs-small-density')

    plt.cla()

    plot(func_large, label=LABELS[2])
    plot(line_large, label=LABELS[3])

    plt.title('Probability Density Distribution (Large Model)')
    plt.xlabel('Probability')
    plt.legend(labels=[LABELS[2], LABELS[3]])

    plt.tight_layout()
    plt.savefig('output/logprobs-large-density')

if __name__ == '__main__':
    func_small = load_data('func_small')
    line_small = load_data('line_small')
    func_large = load_data('func_large')
    line_large = load_data('line_large')

    logprobs_density_plot(func_small, line_small, func_large, line_large)
    cumulative_distribution_function_energy(func_small, line_small, func_large, line_large)
    energy_violinplot(func_small, line_small, func_large, line_large)
    time_violinplot(func_small, line_small, func_large, line_large)
    finish_reason_barplot(func_small, func_large)
    energy_vs_time_scatterplot(func_small, line_small, func_large, line_large)
    distribution_unique_solutions(line_small, line_large)
