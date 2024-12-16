import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import collections
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

from classes import CodeVerificationResult
from classes import GenerationResult, GenerationOutput, GenerationMetrics
from data_processor import DataProcessor

sns.set_theme()
sns.color_palette("colorblind")

RESULTS_DIR = '/Users/thijs/Documents/Git/msc-experiment/results'
TEST_RESULTS_DIR = '/Users/thijs/Documents/Git/msc-experiment/test_results'
SKIPPED_PROBLEMS_FILE = '/Users/thijs/Documents/Git/msc-experiment/tests/skipped_problems.txt'

with open(SKIPPED_PROBLEMS_FILE, 'r') as f:
    skipped_problems = [x.split(' - ')[0] for x in f.readlines()]

def load_test_results(dataset_type, problems):
    results = collections.defaultdict(lambda: collections.defaultdict(list))
    counter = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

    for problem_folder in glob.glob(f'{TEST_RESULTS_DIR}/{dataset_type}/*'):
        problem_id = problem_folder.split('/')[-1]

        if problem_id in skipped_problems:
            continue

        for file in glob.glob(f'{problem_folder}/*.jsonl'):
            line_index = re.search(r'(\d+)\.jsonl', file)[1]

            with open(file) as f:
                for content in f.readlines():
                    result = CodeVerificationResult(**json.loads(content))

                    results[problem_id][line_index].append(result)

    for problem_file in glob.glob(f'{RESULTS_DIR}/**/{dataset_type}.jsonl'):
        problem_id = re.search(r'(\d+)', problem_file)[1]
        problem = problems[int(problem_id)]

        with open(problem_file) as f:
            for line in f.read().splitlines():
                result = GenerationResult(**json.loads(line))
                result.outputs = [GenerationOutput(**o) for o in result.outputs]

                for output, line_prompt in zip(result.outputs, problem.line_prompts):
                    counter[problem_id][line_prompt.line_index][output.text] += 1

    return results, counter

def average_time_histogram(line_small, line_small_counter, line_large, line_large_counter):
    def prepare_data(data, counter):
        average_execution_times = []

        for problem_id, line_results in data.items():
            for line_index, results in line_results.items():
                line_index = int(line_index)

                all_execution_times = sum([[x.time] * counter[problem_id][line_index][x.code] for x in results], [])
                average_execution_time = sum(all_execution_times) / len(all_execution_times)

                average_execution_times.append(average_execution_time)

        return average_execution_times

    data_small = prepare_data(line_small, line_small_counter)
    data_large = prepare_data(line_large, line_large_counter)

    df_small = pd.DataFrame({'Average Execution Time': data_small, 'Model': 'Small Model (1.5B)'})
    df_large = pd.DataFrame({'Average Execution Time': data_large, 'Model': 'Large Model (9B)'})
    
    df_combined = pd.concat([df_small, df_large])

    plt.figure(figsize=(8, 4))
    sns.histplot(df_combined, x='Average Execution Time', hue='Model', multiple="dodge", bins=20, kde=False)

    plt.title('Average Test Execution Time per Line Generation')
    plt.xlabel('Execution Time (s)')

    plt.tight_layout()
    plt.savefig('output/average-test-execution-time-histogram')
    plt.cla()

def average_test_pass_percentages_histogram(line_small, line_small_counter, line_large, line_large_counter):
    def prepare_data(data, counter):
        average_test_pass_percentages = []

        for problem_id, line_results in data.items():
            for line_index, results in line_results.items():
                line_index = int(line_index)

                all_pass_percentages = sum([[int(x.tests_passed)] * counter[problem_id][line_index][x.code] for x in results], [])
                average_pass_percentage = sum(all_pass_percentages) / len(all_pass_percentages) * 100

                average_test_pass_percentages.append(average_pass_percentage)

        return average_test_pass_percentages

    data_small = prepare_data(line_small, line_small_counter)
    data_large = prepare_data(line_large, line_large_counter)

    df_small = pd.DataFrame({'Pass Percentage': data_small, 'Model': 'Small Model (1.5B)'})
    df_large = pd.DataFrame({'Pass Percentage': data_large, 'Model': 'Large Model (9B)'})
    
    df_combined = pd.concat([df_small, df_large])

    plt.figure(figsize=(8, 4))
    sns.histplot(df_combined, x='Pass Percentage', hue='Model', multiple="dodge", bins=20, kde=False)

    plt.title('Average Test Pass Percentage per Line Generation')
    plt.xlabel('Pass Percentage (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()

    plt.savefig('output/average-test-pass-percentage-per-line-histogram')
    plt.cla()

def average_pass_percentage_per_line_violinplot(line_small, line_large):
    fig, ax = plt.subplots(figsize=(8,4))
    palette = sns.color_palette()

    def plot(data, label, color):
        pass_percentages = []
        for problem_id, line_results in data.items():
            for line_index, results in line_results.items():
                all_pass_percentages = [int(x.tests_passed) for x in results]
                pass_percentages.append(sum(all_pass_percentages) / len(all_pass_percentages) * 100)

        df = pd.DataFrame({'Value': pass_percentages, 'Category': [label] * len(pass_percentages)})
        sns.violinplot(x='Category', y='Value', data=df, ax=ax, inner=None, color=color)

    plot(line_small, label='Line-level, Small Model', color=palette[0])
    plot(line_large, label='Line-level, Large Model', color=palette[1])

    plt.title('Line-level Test Pass Percentage Distribution')
    plt.tight_layout()
    plt.savefig('output/test-pass-percentage-per-line-violinplot')
    plt.cla()

def pass_percentage_small_line_vs_large_line_scatterplot(line_small, line_large):
    small_pass = []
    large_pass = []

    for problem_id, small_line_results in line_small.items():
        if problem_id not in line_large:
            continue

        large_line_results = line_large[problem_id]

        for line_index, small_results in small_line_results.items():
            if line_index not in large_line_results:
                continue

            large_results = large_line_results[line_index]

            small_pass_percentages = [int(x.tests_passed) for x in small_results]
            large_pass_percentages = [int(x.tests_passed) for x in large_results]

            average_small_pass_percentage = sum(small_pass_percentages) / len(small_pass_percentages)
            average_large_pass_percentage = sum(large_pass_percentages) / len(large_pass_percentages)

            small_pass.append(average_small_pass_percentage)
            large_pass.append(average_large_pass_percentage)

    sns.scatterplot(x=small_pass, y=large_pass)

    plt.title('Line-level Generation Test Pass Percentage')
    plt.xlabel('Small Model')
    plt.ylabel('Large Model')

    plt.tight_layout()
    plt.savefig('output/small-vs-large-line-level-pass-percentage-scatterplot')
    plt.cla()

if __name__ == '__main__':
    """
    line_small = load_test_results('line_small')

    all_combinations = []

    for problem_id, line_results in line_small.items():
        combinations = 1
        times = []

        for generations in line_results.values():
            tests_passed = sum([int(x.tests_passed) for x in generations]) 

            if tests_passed > 0:
                combinations *= tests_passed
                times.extend([x.time for x in generations if x.tests_passed])

        if times:
            average_time = sum(times) / len(times)
            all_combinations.append((problem_id, combinations, average_time))

    total_times = np.log10([c*t for _,c,t in all_combinations])

    sns.histplot(data=total_times)
    plt.show()

    #all_combinations_sorted = sorted(all_combinations, key=lambda x: x[1] * x[2])

    #for problem_id, combinations, average_time in all_combinations_sorted:
    #    print(f'Problem {problem_id}:\t{combinations} ({combinations * average_time:2f}s)')
    """

    problems = DataProcessor.load(input_file_path='/Users/thijs/Documents/Git/msc-experiment/data/test-dataset.jsonl')

    line_small, line_small_counter = load_test_results('line_small', problems)
    line_large, line_large_counter = load_test_results('line_large', problems)

    average_time_histogram(line_small, line_small_counter, line_large, line_large_counter)
    #average_test_pass_percentages_histogram(line_small, line_small_counter, line_large, line_large_counter)

    """
    line_large = load_test_results('line_large')
   
    average_pass_percentage_per_line_violinplot(line_small, line_large)
    pass_percentage_small_line_vs_large_line_scatterplot(line_small, line_large)
    average_time_histogram(line_small)
    average_test_pass_percentages_histogram(line_small, line_large)
    """
