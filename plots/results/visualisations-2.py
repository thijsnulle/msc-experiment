import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import collections
import glob
import json
import math
import numpy as np
import pandas as pd
import re

from classes import MCSResult, CodeVerificationResult, DataclassJSONEncoder
from classes import GenerationMetrics, GenerationOutput, GenerationResult
from data_processor import DataProcessor
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.color_palette("colorblind")

RESULTS_FOLDER = Path('/Users/thijsnulle/Documents/Git/msc-experiment/results')
TEST_RESULTS_FOLDER = Path('/Users/thijsnulle/Documents/Git/msc-experiment/test_results')
MCS_RESULTS_FOLDER = Path('/Users/thijsnulle/Documents/Git/msc-experiment/mcs_results')

LINE_LARGE_FOLDER = MCS_RESULTS_FOLDER / 'line_large'
LINE_SMALL_FOLDER = MCS_RESULTS_FOLDER / 'line_small'

COMPRESSED_LINE_LARGE_FOLDER = MCS_RESULTS_FOLDER / '_compressed/line_large'
COMPRESSED_LINE_SMALL_FOLDER = MCS_RESULTS_FOLDER / '_compressed/line_small'

import itertools

def get_excess_tokens_for_func_level(code_lines, tokens):
    code_tokens_length = len(tokens)
    code_lines_until_return = list(itertools.dropwhile(
        lambda x: not x.strip().startswith(('return', 'pass')),
        reversed(code_lines),
    ))
    code_lines_until_return.reverse()


    if len(code_lines_until_return) == 0:
        return None

    # Excess tokens after function implementation
    excess_tokens = 0
    if len(code_lines) != len(code_lines_until_return):
        newline_counter = len(code_lines) - len(code_lines_until_return)

        while newline_counter > 0:
            if tokens[-1] == '\n':
                newline_counter -= 1

            tokens.pop()

        excess_tokens += (code_tokens_length - len(tokens))

    # Excess print-statement tokens
    print_indexes = [i for i, t in enumerate(tokens) if t == 'print']

    for i in print_indexes:
        preceding_whitespace_tokens = list(itertools.takewhile(
            lambda x: all(y == ' ' for y in x),
            reversed(tokens[:i]),
        ))

        succeeding_print_tokens = list(itertools.takewhile(
            lambda x: x != '\n',
            tokens[i:],
        ))

        excess_tokens += len(preceding_whitespace_tokens)
        excess_tokens += len(succeeding_print_tokens)

    # Excess comment tokens
    comments_indexes = [i for i, t in enumerate(tokens) if t == '#']

    for i in comments_indexes:
        preceding_whitespace_tokens = list(itertools.takewhile(
            lambda x: all(y == ' ' for y in x),
            reversed(tokens[:i]),
        ))

        succeeding_comment_tokens = list(itertools.takewhile(
            lambda x: x != '\n',
            tokens[i:],
        ))

        excess_tokens += len(preceding_whitespace_tokens)
        excess_tokens += len(succeeding_comment_tokens)

    return excess_tokens

def get_excess_tokens_for_line_level(tokens):
    if '#' not in tokens:
        return 0

    excess_tokens = 0

    # Excess print-statement tokens
    print_indexes = [i for i, t in enumerate(tokens) if t == 'print']

    for i in print_indexes:
        preceding_whitespace_tokens = list(itertools.takewhile(
            lambda x: all(y == ' ' for y in x),
            reversed(tokens[:i]),
        ))

        succeeding_print_tokens = list(itertools.takewhile(
            lambda x: x != '\n',
            tokens[i:],
        ))

        excess_tokens += len(preceding_whitespace_tokens)
        excess_tokens += len(succeeding_print_tokens)


    # Excess comment tokens
    comments_indexes = [i for i, t in enumerate(tokens) if t == '#']

    for i in comments_indexes:
        preceding_whitespace_tokens = list(itertools.takewhile(
            lambda x: all(y == ' ' for y in x),
            reversed(tokens[:i]),
        ))

        succeeding_comment_tokens = list(itertools.takewhile(
            lambda x: x != '\n',
            tokens[i:],
        ))

        excess_tokens += len(preceding_whitespace_tokens)
        excess_tokens += len(succeeding_comment_tokens)

    return excess_tokens

def wrong_solutions():
    def get_energies(type):
        mean_wasted_energies = []
        mean_wasted_energy_percentages = []
        median_wasted_energies = []
        median_wasted_energy_percentages = []

        for problem_id in map(str, range(1140)):
            print(problem_id)

            with open('/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt', 'r') as f:
                if any(l.startswith(f'{problem_id} - ') for l in f.readlines()):
                    continue

            if 'func' in type:
                with open(glob.glob(f'{TEST_RESULTS_FOLDER}/{type}/{problem_id}/*')[0], 'r') as f:
                    func_large_test_results = [CodeVerificationResult(**json.loads(c)) for c in f.readlines()]

                with open(f'{RESULTS_FOLDER}/{problem_id}/{type}.jsonl') as f:
                    results = []

                    for content in f.readlines():
                        result = GenerationResult(**json.loads(content))
                        result.outputs = [GenerationOutput(**o) for o in result.outputs]
                        result.metrics = GenerationMetrics(**result.metrics)

                        results.append(result)

                total_energy = 0
                wasted_energies = []

                if all(not t.tests_passed for t in func_large_test_results):
                    continue

                for test_result in func_large_test_results:
                    try:
                        result = next(filter(lambda r: r.outputs[0].text.startswith(test_result.code), results))
                    except StopIteration:
                        continue

                    tokens = result.outputs[0].tokens
                    total_tokens = len(tokens)
                    code_lines = ''.join(tokens).splitlines()

                    excess_tokens = get_excess_tokens_for_func_level(code_lines, tokens)

                    total_tokens -= (excess_tokens or 0)
                    total_energy += total_tokens * result.metrics.energy_per_token

                    if not test_result.tests_passed or excess_tokens is None:
                        wasted_energies.append(total_tokens * result.metrics.energy_per_token)

            elif 'line' in type:
                with open(f'{MCS_RESULTS_FOLDER}/{type}/{problem_id}.jsonl') as f:
                    mcs_results = []

                    for content in f.readlines():
                        mcs_result = MCSResult(**json.loads(content))
                        mcs_result.result = CodeVerificationResult(**mcs_result.result)
                        mcs_results.append(mcs_result)

                all_lines = list(sorted(set(itertools.chain(*[r.selected_lines for r in mcs_results]))))
                selected_mcs_results = list(filter(lambda r: len(r.selected_lines) == max(len(all_lines) - 1, 1), mcs_results))
                selected_lines = sorted(selected_mcs_results[0].selected_lines)

                with open(f'{RESULTS_FOLDER}/{problem_id}/{type}.jsonl') as f:
                    results = []

                    for content in f.readlines():
                        result = GenerationResult(**json.loads(content))
                        result.outputs = [GenerationOutput(**o) for o in result.outputs]
                        result.metrics = GenerationMetrics(**result.metrics)

                        results.append(result)

                total_energy = 0
                wasted_energies = []

                for mcs_result in selected_mcs_results:
                    code_lines = mcs_result.result.code.splitlines()

                    energies = []
                    for line in sorted(mcs_result.selected_lines):
                        try:
                            i = all_lines.index(line)
                            result = next(filter(lambda r: code_lines[line].endswith(r.outputs[i].text), results))

                            energies.append(len(result.outputs[i].tokens) * result.metrics.energy_per_token)
                        except StopIteration:
                            continue

                    total_energy += sum(energies)
                    
                    if not mcs_result.result.tests_passed:
                        wasted_energies.extend(energies)

            if total_energy and wasted_energies:
                mean_wasted_energies.append(sum(wasted_energies) / len(wasted_energies))
                mean_wasted_energy_percentages.append(sum(wasted_energies) / total_energy)
                median_wasted_energies.append(wasted_energies[len(wasted_energies) // 2])
                median_wasted_energy_percentages.append(wasted_energies[len(wasted_energies) // 2] / total_energy)

        return mean_wasted_energies, mean_wasted_energy_percentages, median_wasted_energies, median_wasted_energy_percentages

    line_small_energies = get_energies('line_small')
    line_large_energies = get_energies('line_large')
    func_small_energies = get_energies('func_small')
    func_large_energies = get_energies('func_large')

    data = pd.DataFrame({
        "Values": line_small_energies[0] + line_large_energies[0] + func_small_energies[0] + func_large_energies[0],
        "Category": 
            ["Line-level, Small Model"] * len(line_small_energies[0]) +
            ["Line-level, Large Model"] * len(line_large_energies[0]) +
            ["Function-level, Small Model"] * len(func_small_energies[0]) +
            ["Function-level, Large Model"] * len(func_large_energies[0]),
    })

    sns.boxplot(x="Category", y="Values", data=data)

    plt.title("Comparison of Two Lists")
    plt.xlabel("Category")
    plt.ylabel("Values")
    plt.show()

def excess_tokens():
    def get_excess_tokens_for_file(filename):
        with open(f'{RESULTS_FOLDER}/{problem_id}/{filename}') as f:
            excess_tokens_list = []

            for content in f.readlines():
                result = GenerationResult(**json.loads(content))
                result.outputs = [GenerationOutput(**o) for o in result.outputs]
                result.metrics = GenerationMetrics(**result.metrics)

                excess_tokens = 0

                for output in result.outputs:
                    if not output.text or output.finish_reason == 'finish':
                        continue

                    code_lines = ''.join(output.tokens).splitlines()
                    code_tokens_length = len(output.tokens)

                    if len(code_lines) == 1:
                        excess_tokens = get_excess_tokens_for_line_level(output.tokens)
                    else:
                        excess_tokens = get_excess_tokens_for_func_level(code_lines, output.tokens)

                    if excess_tokens is not None:
                        excess_tokens_list.append(excess_tokens / code_tokens_length)

        if len(excess_tokens_list) == 0:
            return None

        return sum(excess_tokens_list) / len(excess_tokens_list)

    excess_tokens_lists = collections.defaultdict(list)

    for problem_id in map(str, range(1140)):
        with open('/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt', 'r') as f:
            if any(l.startswith(f'{problem_id} - ') for l in f.readlines()):
                continue

        if x := get_excess_tokens_for_file('line_large.jsonl'): excess_tokens_lists['Line-level, Large Model'].append(x)
        if x := get_excess_tokens_for_file('func_large.jsonl'): excess_tokens_lists['Function-level, Large Model'].append(x)
        if x := get_excess_tokens_for_file('line_small.jsonl'): excess_tokens_lists['Line-level, Small Model'].append(x)
        if x := get_excess_tokens_for_file('func_small.jsonl'): excess_tokens_lists['Function-level, Small Model'].append(x)

    import matplotlib.pyplot as plt

    data = []
    for category, values in excess_tokens_lists.items():
        for value in values:
            data.append({'Category': category, 'Excess Tokens': value})

    df = pd.DataFrame(data)

    # Create subsets for the second graph
    line_level_data = df[df['Category'].isin(['Line-level, Small Model', 'Line-level, Large Model'])]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [2, 1]})

    _palette = sns.color_palette()[:4]
    palette = [_palette[2], _palette[0], _palette[3], _palette[1]]

    ax = sns.boxplot(ax=axes[0], x='Category', y='Excess Tokens', hue='Category', data=df, width=0.33, showfliers=False, palette=palette)
    axes[0].set_title('Proportion of Excess Tokens in Generated Outputs')
    axes[0].set_xlabel(None)
    axes[0].set_ylabel('Proportion of Excess Tokens')
    axes[0].set_xticks([])
    axes[0].legend(labels=excess_tokens_lists.keys(), loc='upper right')

    palette = palette[2:]

    sns.boxplot(ax=axes[1], x='Category', y='Excess Tokens', hue='Category', data=line_level_data, width=0.33, showfliers=False, palette=palette)
    axes[1].set_title('Line-level Models')
    axes[1].set_xlabel(None)
    axes[1].set_ylabel('Proportion of Excess Tokens')
    axes[1].set_xticks([])

    plt.tight_layout()
    plt.savefig('output/proportion-of-excess-tokens', dpi=500)

    return

    import scipy.stats as stats

    def calculate_p_value(list1, list2):
        t_stat, p_value = stats.ttest_ind(list1, list2)
        return t_stat, p_value

    def calculate_p_value_nonparametric(list1, list2):
        u_stat, p_value = stats.mannwhitneyu(list1, list2, alternative='two-sided')
        return u_stat, p_value

    def is_normal_distribution(data, alpha=0.05):
        stat, p_value = stats.shapiro(data)
        return p_value > alpha, p_value

    print('Normal Distribution')
    print('Function-level, Small Model', is_normal_distribution(excess_tokens_lists['Function-level, Small Model']))
    print('Function-level, Large Model', is_normal_distribution(excess_tokens_lists['Function-level, Large Model']))
    print('Line-level, Small Model', is_normal_distribution(excess_tokens_lists['Line-level, Small Model']))
    print('Line-level, Large Model', is_normal_distribution(excess_tokens_lists['Line-level, Large Model']))
    print()

    print('U-stat, P-value')
    print('Small Model, function-level vs line-level', calculate_p_value_nonparametric(excess_tokens_lists['Function-level, Small Model'], excess_tokens_lists['Line-level, Small Model']))
    print('Large Model, function-level vs line-level', calculate_p_value_nonparametric(excess_tokens_lists['Function-level, Large Model'], excess_tokens_lists['Line-level, Large Model']))
    print('Function-level, small model vs large model', calculate_p_value_nonparametric(excess_tokens_lists['Function-level, Small Model'], excess_tokens_lists['Function-level, Large Model']))
    print('Line-level, small model vs large model', calculate_p_value_nonparametric(excess_tokens_lists['Line-level, Small Model'], excess_tokens_lists['Line-level, Large Model']))


def distribution_small_model_outperforms_large_model():
    """
    Displays the distribution of scenarios where a small model with line-level
    completions outperforms a large model with function-level completions. The
    X-axis represents the percentage of lines the small model can maximally
    substitute while still outperforming the large model.

    It also prints for how many of the problems the small model never
    outperforms the large model.
    """

    problems = DataProcessor.load('/Users/thijsnulle/Documents/Git/msc-experiment/data/processed-dataset.jsonl')

    small_averages = []
    large_averages = []

    bin_size = 0.1
    bin_results = collections.defaultdict(list)

    for problem_id in map(str, range(1140)):
        print(problem_id)
        with open('/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt', 'r') as f:
            if any(l.startswith(f'{problem_id} - ') for l in f.readlines()):
                continue

        with open(f'{RESULTS_FOLDER}/{problem_id}/line_small.jsonl') as f:
            problem = problems[int(problem_id)]
            generations = collections.defaultdict(lambda: collections.defaultdict(list))

            for content in f.readlines():
                result = GenerationResult(**json.loads(content))
                result.outputs = [GenerationOutput(**o) for o in result.outputs]
                result.metrics = GenerationMetrics(**result.metrics)

                for prompt, output in zip(problem.line_prompts, result.outputs):
                    generations[prompt.line_index][output.text].append(len(output.tokens) * result.metrics.energy_per_token)

            generations = {
                line_index: { text: sum(energy) / len(energy) for text, energy in energies.items() }
                for line_index, energies in generations.items()
            }

        with open(f'{RESULTS_FOLDER}/{problem_id}/func_large.jsonl') as f:
            func_large_energies = []

            for content in f.readlines():
                result = GenerationResult(**json.loads(content))
                result.metrics = GenerationMetrics(**result.metrics)

                func_large_energies.append(result.metrics.energy)

        # Get the large model function-level accuracy
        with open(glob.glob(f'{TEST_RESULTS_FOLDER}/func_large/{problem_id}/*')[0], 'r') as f:
            func_large_test_results = [CodeVerificationResult(**json.loads(c)).tests_passed for c in f.readlines()]
            func_large_test_accuracy = sum(func_large_test_results) / len(func_large_test_results)

        # Get the small model line-level accuracies and energy per line-level generation
        line_small_test_results = collections.defaultdict(list)

        with open(f'{LINE_SMALL_FOLDER}/{problem_id}.jsonl', 'r') as f:
            for content in f.readlines():
                mcs_result = MCSResult(**json.loads(content))
                mcs_result.result = CodeVerificationResult(**mcs_result.result)

                lines = mcs_result.result.code.splitlines()

                total_energy = 0
                for line in mcs_result.selected_lines:
                    if line not in generations:
                        continue

                    energies = generations[line]

                    for code, energy in energies.items():
                        if not lines[line].endswith(code):
                            continue

                        total_energy += energy

                line_small_test_results[len(mcs_result.selected_lines)].append((mcs_result.result.tests_passed, total_energy))

        average_func_large_energy = sum(func_large_energies) / len(func_large_energies)

        num = 0
        total_lines = len(line_small_test_results)

        for num_lines, test_results in sorted(line_small_test_results.items(), reverse=True):
            tests = [x[0] for x in test_results]
            energies = [x[1] for x in test_results]

            if len(energies) == 0 or sum(energies) == 0:
                continue

            energy = sum(energies) / len(energies)
            percentage_change = (energy - average_func_large_energy) / average_func_large_energy * 100

            bin = round(max(bin_size * round((num_lines / total_lines) / bin_size), bin_size), 1)
            #bin_results[bin].append(1 / (sum(energies) / len(energies) / average_func_large_energy))
            bin_results[bin].append(percentage_change)

            if sum(tests) / len(tests) >= func_large_test_accuracy and num == 0:
                num = num_lines
                energy = sum(energies) / len(energies)

    # Prepare data for boxplot
    sorted_bins = sorted(bin_results.items())  # Sort by bin (key)

    # Prepare data for boxplot
    results_list = [results for _, results in sorted_bins]
    bin_labels = [bin_label for bin_label, _ in sorted_bins]

    # Plot boxplot
    sns.boxplot(data=results_list, showfliers=False)
    plt.xlabel('Fraction of Generated Lines')
    plt.ylabel('Reduction in Carbon Emissions (%)')
    plt.title('Reduction in Carbon Emissions vs Fraction of Generated Lines')
    plt.xticks(range(len(bin_labels)), bin_labels)  # Optional: rotate x-axis labels if needed
    plt.tight_layout()
    plt.savefig('output/-decrease-carbon-emissions-vs-fraction-lines-substituted', dpi=500)
    plt.clf()

    # Filter bins to only keep those between 0.5 and 1.0
    filtered_bins = {bin_label: results for bin_label, results in sorted_bins if 0.5 <= bin_label <= 1.0}

    filtered_results_list = [results for _, results in filtered_bins.items()]
    filtered_bin_labels = [str(bin_label) for bin_label in filtered_bins.keys()]

    palette = sns.color_palette()[4:]
    
    sns.boxplot(data=filtered_results_list, showfliers=False, palette=palette)
    plt.xlabel('Fraction of Generated Lines')
    plt.ylabel('Reduction in Carbon Emissions (%)')
    plt.title('Reduction in Carbon Emissions for 0.5 ≤ Fraction ≤ 1.0')
    plt.xticks(range(len(filtered_bin_labels)), filtered_bin_labels)  # Optional: rotate x-axis labels if needed
    plt.tight_layout()
    plt.savefig('output/-decrease-carbon-emissions-vs-fraction-lines-substituted-zoom', dpi=500)


def compress(folder, compressed_folder):
    for filename in glob.glob(f'{folder}/*'):
        problem_id = re.search(r'(\d+)\.jsonl', filename)[1]
    
        mcs_results_counter = collections.Counter()

        with open(filename, 'r') as f:
            for content in f.readlines():
                mcs_result = MCSResult(**json.loads(content))
                mcs_result.result = CodeVerificationResult(**mcs_result.result)

                mcs_results_counter[mcs_result] += 1

        with open(compressed_folder / f'{problem_id}.csv', 'w+') as f:
            print('COUNT,RESULT', file=f)

            for result, count in mcs_results_counter.most_common():
                print(f'{count},{json.dumps(result,cls=DataclassJSONEncoder)}', file=f)


def load(compressed_folder):
    def load_data(compressed_folder):
        fraction_results = collections.defaultdict(lambda: collections.defaultdict(list))

        for filename in glob.glob(f'{compressed_folder}/*'):
            problem_id = re.search(r'(\d+)\.csv', filename)[1]
            mcs_results_counter = collections.Counter()
     
            with open(filename, 'r') as f:
                for _content in f.readlines()[1:]:
                    count, content = _content.split(',', maxsplit=1)

                    mcs_result = MCSResult(**json.loads(content))
                    mcs_result.result = CodeVerificationResult(**mcs_result.result)

                    mcs_results_counter[mcs_result] = int(count)

            correct_counter = collections.Counter()
            total_counter = collections.Counter()

            max_lines = 0

            for result, cnt in mcs_results_counter.most_common()[::-1]:
                num_lines = len(result.selected_lines)
                max_lines = max(num_lines, max_lines)

                total_counter[num_lines] += cnt

                if result.result.tests_passed:
                    correct_counter[num_lines] += cnt

            for num_lines in total_counter.keys():
                fraction = math.ceil(num_lines / max_lines * 10) / 10

                fraction_results[problem_id][fraction].append(correct_counter[num_lines] / total_counter[num_lines])

        return fraction_results

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    small_fraction_results = load_data(COMPRESSED_LINE_SMALL_FOLDER)
    large_fraction_results = load_data(COMPRESSED_LINE_LARGE_FOLDER)

    data = []

    for problem_id, results in small_fraction_results.items():
        fractions, results_group = zip(*sorted(results.items()))

        fractions = list(fractions)
        results_group = list(results_group)

        for fraction, results in zip(fractions, results_group):
            data.extend([(fraction, result, 'Small Model') for result in results])

    df = pd.DataFrame(data, columns=['Fraction', 'Result', 'Group'])

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Fraction', y='Result', hue='Group', data=df, order=sorted(fractions))

    plt.title('Test Pass Percentage per Fraction of Line-level Completions')
    plt.xlabel('Fraction')
    plt.ylabel('Test Pass Percentage')

    plt.show()

    return

    sorted_fractions = sorted(fractions)
    sorted_results = [results_list[fractions.index(f)] for f in sorted_fractions]

    plt.figure(figsize=(10,6))
    plt.boxplot(sorted_results, labels=sorted_fractions)

def should_skip(problem_id):
    with open('/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt', 'r') as f:
        return any(l.startswith(f'{problem_id} - ') for l in f.readlines())

def plot_test_pass_accuracy():
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    large_solution_lt_10 = [None, 0.86, None, 0.79, None, 0.69, None, 0.59, None, 0.53]
    large_solution_gte_10 = [0.86, 0.73, 0.61, 0.52, 0.44, 0.38, 0.33, 0.28, 0.25, 0.21]

    small_solution_lt_10 = [None, 0.79, None, 0.69, None, 0.58, None, 0.46, None, 0.42]
    small_solution_gte_10 = [0.79, 0.64, 0.51, 0.42, 0.34, 0.28, 0.28, 0.19, 0.17, 0.13]

    large_solution_lt_10 = [x if x is not None else np.nan for x in large_solution_lt_10]
    small_solution_lt_10 = [x if x is not None else np.nan for x in small_solution_lt_10]

    data = {
        'Fraction': fractions * 4,
        'Test Pass Accuracy': (
            large_solution_lt_10 + large_solution_gte_10 +
            small_solution_lt_10 + small_solution_gte_10
        ),
        'Solution': 
            ['Large Model — Length < 10'] * 10 + 
            ['Large Model — Length ≥ 10'] * 10 + 
            ['Small Model — Length < 10'] * 10 + 
            ['Small Model — Length ≥ 10'] * 10,
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='Fraction', y='Test Pass Accuracy', hue='Solution', marker='o')

    plt.xlabel('Fraction of Lines Substituted')
    plt.ylabel('Test Pass Accuracy')
    plt.title('MCS Test Results by Fraction of Lines Substituted (Large vs Small Models)')
    plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('output/test-pass-accuracy-line-level', dpi=500)

def fraction_where_small_model_outperforms_large_model():
    problems = DataProcessor.load('../../data/processed-dataset.jsonl')

    results = collections.Counter()

    for problem_id in map(str, range(1140)):
        if should_skip(problem_id):
            continue

        max_lines = len(problems[int(problem_id)].line_prompts) - 1

        with open(glob.glob(f'{TEST_RESULTS_FOLDER}/func_large/{problem_id}/*')[0]) as f:
            func_large_test_results = [CodeVerificationResult(**json.loads(c)).tests_passed for c in f.readlines()]

        func_large_accuracy = sum(func_large_test_results) / len(func_large_test_results)

        line_small_test_results = collections.defaultdict(list)
        with open(f'{MCS_RESULTS_FOLDER}/line_small/{problem_id}.jsonl') as f:
            for content in f.readlines():
                result = MCSResult(**json.loads(content))

                line_small_test_results[len(result.selected_lines)].append(CodeVerificationResult(**result.result).tests_passed)

        line_small_accuracies = { key: sum(xs) / len(xs) for key, xs in line_small_test_results.items() }

        found = False
        for key, accuracy in sorted(line_small_accuracies.items(), reverse=True):
            if accuracy < func_large_accuracy:
                continue

            found = True
            break

        if not found:
            results[0] += 1

            continue

        bin = round(math.ceil(key / max_lines / 0.2) * 0.2, 1)

        results[bin] += 1

    total = sum(c for _, c in results.most_common())

    for key, count in results.most_common():
        print(key, count, f'({count/total:.2f}%)')

def plot_small_outperforms_large():
    small_line_small_bin = {0:227, 0.1:19, 0.2:111, 0.3:85, 0.4:88, 0.5:75, 0.6:42, 0.7:38, 0.8:34, 0.9:23, 1.0:292}
    small_line_large_bin = {0:227, 0.2:130, 0.4:173, 0.6:117, 0.8:72, 1.0:315}
    large_line_small_bin = {0:140, 0.1:14, 0.2:75, 0.3:67, 0.4:77, 0.5:83, 0.6:49, 0.7:43, 0.8:46, 0.9:20, 1.0:420}
    large_line_large_bin = {0:140, 0.2:89, 0.4:144, 0.6:132, 0.8:89, 1.0:440}

    bins = sorted(small_line_small_bin.keys())
    small_small_vals = [small_line_small_bin.get(b, 0) for b in bins]
    small_large_vals = [small_line_large_bin.get(b, 0) for b in bins]
    large_small_vals = [large_line_small_bin.get(b, 0) for b in bins]
    large_large_vals = [large_line_large_bin.get(b, 0) for b in bins]

    data = []
    #for bin_key, count in small_line_small_bin.items():
    #    data.append({'Fraction Substituted': bin_key, 'Outperformance Count': count, 'Group': 'Small Model'})
    for bin_key, count in small_line_large_bin.items():
        data.append({'Fraction Substituted': bin_key, 'Outperformance Count': count, 'Group': 'Small Model'})
    #for bin_key, count in large_line_small_bin.items():
    #    data.append({'Fraction Substituted': bin_key, 'Outperformance Count': count, 'Group': 'Large Line, Small Bin'})
    for bin_key, count in large_line_large_bin.items():
        data.append({'Fraction Substituted': bin_key, 'Outperformance Count': count, 'Group': 'Large Model'})

    df = pd.DataFrame(data)

    print(df)

    plt.figure(figsize=(8,5))
    sns.barplot(x='Fraction Substituted', y='Outperformance Count', hue='Group', data=df)

    plt.xlabel('Fraction of Lines Substituted')
    plt.ylabel('Frequency')
    plt.title('Fraction of Line-Level Completions Where Function-Level Completions Are Outperformed')
    plt.grid(True)
    plt.legend(title='Model Size')
    plt.tight_layout()
    plt.savefig('output/line-level-outperforms-func-level', dpi=500)


def test_pass_accuracy_line_level():
    problems = DataProcessor.load('../../data/processed-dataset.jsonl')

    test_results_small = collections.defaultdict(list)
    test_results_large = collections.defaultdict(list)

    for problem_id in map(str, range(1140)):
        if should_skip(problem_id):
            continue

        num_lines = max(len(problems[int(problem_id)].line_prompts) - 1, 1)

        mcs_results = []

        with open(f'{MCS_RESULTS_FOLDER}/line_small/{problem_id}.jsonl') as f:
            for content in f.readlines():
                mcs_result = MCSResult(**json.loads(content))
                mcs_result.result = CodeVerificationResult(**mcs_result.result)
                mcs_results.append(mcs_result)

        with open(f'{MCS_RESULTS_FOLDER}/line_small/{problem_id}.jsonl') as f:
            for content in f.readlines():
                mcs_result = MCSResult(**json.loads(content))
                mcs_result.result = CodeVerificationResult(**mcs_result.result)
                mcs_results.append(mcs_result)

        for result in mcs_results:
            bin_size = 0.1 if num_lines < 10 else 0.1
            bin = round(math.ceil(len(result.selected_lines) / num_lines / bin_size) * bin_size, 1)

            if num_lines < 10:
                test_results_small[bin].append(result.result.tests_passed)
            else:
                test_results_large[bin].append(result.result.tests_passed)

    print('Solution Length < 10')
    for bin, test_results in sorted(test_results_small.items()):
        print(bin, f'{sum(test_results) / len(test_results):.2f}')

    print()
    print('Solution Length >= 10')
    for bin, test_results in sorted(test_results_large.items()):
        print(bin, f'{sum(test_results) / len(test_results):.2f}')


def reduction_carbon_emissions():
    problems = DataProcessor.load('../../data/processed-dataset.jsonl')

    energy_results = collections.defaultdict(lambda: collections.defaultdict(list))
    energy_results_better = collections.defaultdict(list)

    for problem_id in map(str, range(1140)):
        print(problem_id)
        if should_skip(problem_id):
            continue

        energy_per_line = dict()

        for _type in ['line_small', 'line_large']:
            with open(f'{RESULTS_FOLDER}/{problem_id}/{_type}.jsonl') as f:
                problem = problems[int(problem_id)]
                energies = collections.defaultdict(lambda: collections.defaultdict(list))

                for content in f.readlines():
                    result = GenerationResult(**json.loads(content))
                    result.outputs = [GenerationOutput(**o) for o in result.outputs]
                    result.metrics = GenerationMetrics(**result.metrics)

                    for prompt, output in zip(problem.line_prompts, result.outputs):
                        energies[prompt.line_index][output.text].append(len(output.tokens) * result.metrics.energy_per_token)

                energy_per_line[_type] = {
                    line_index: { text: sum(energy) / len(energy) for text, energy in _energies.items() }
                    for line_index, _energies in energies.items()
                }

        with open(f'{RESULTS_FOLDER}/{problem_id}/func_large.jsonl') as f:
            func_large_energies = [GenerationResult(**json.loads(c)).metrics['energy'] for c in f.readlines()]
            func_large_energy = sum(func_large_energies) / len(func_large_energies)

        with open(glob.glob(f'{TEST_RESULTS_FOLDER}/func_large/{problem_id}/*')[0], 'r') as f:
            func_large_test_results = [CodeVerificationResult(**json.loads(c)).tests_passed for c in f.readlines()]
            func_large_test_accuracy = sum(func_large_test_results) / len(func_large_test_results)

        num_prompts = len(problem.line_prompts)

        bin_size = 0.1

        for folder in [LINE_SMALL_FOLDER, LINE_LARGE_FOLDER]:
            _type = 'line_small' if folder == LINE_SMALL_FOLDER else 'line_large'

            with open(f'{folder}/{problem_id}.jsonl', 'r') as f:
                mcs_results = []
                mcs_test_results = collections.defaultdict(list)

                for content in f.readlines():
                    mcs_result = MCSResult(**json.loads(content))
                    mcs_result.result = CodeVerificationResult(**mcs_result.result)

                    mcs_results.append(mcs_result)

                    percentage = len(mcs_result.selected_lines) / max(num_prompts - 1, 1)
                    bin = round(math.ceil(percentage / bin_size) * bin_size, 1)

                    mcs_test_results[bin].append(mcs_result.result.tests_passed)

                try:
                    selected_bin = next(filter(lambda x: sum(x[1]) / len(x[1]) >= func_large_test_accuracy, mcs_test_results.items()))[0]
                except StopIteration:
                    selected_bin = None

                for mcs_result in mcs_results:
                    percentage = len(mcs_result.selected_lines) / max(num_prompts - 1, 1)
                    bin = round(math.ceil(percentage / bin_size) * bin_size, 1)

                    lines = mcs_result.result.code.splitlines()

                    total_energy = 0

                    for line_index in mcs_result.selected_lines:
                        if line_index not in energy_per_line[_type]:
                            continue

                        try:
                            key = next(filter(lambda x: lines[line_index].endswith(x), energy_per_line[_type][line_index].keys()))
                        except StopIteration:
                            continue

                        total_energy += energy_per_line[_type][line_index][key]

                    if not total_energy:
                        continue

                    energy_results[_type][bin].append(1 / (total_energy / func_large_energy))
                    
                    if bin == selected_bin:
                        energy_results_better[_type].append(1 / (total_energy / func_large_energy))

    plt.figure()

    def format_yticks(x, pos):
        return f'{int(x)}x'
 
    df = pd.DataFrame.from_dict(energy_results_better, orient='index').T
    df_melted = df.melt(var_name="Model", value_name="Reduction")

    sns.boxplot(data=df_melted, x="Model", y="Reduction", showfliers=False, width=0.33, label='Small Model' if _type == 'line_small' else 'Large Model')

    plt.title('Carbon Emission Reduction: Small Model with Line-level Completions\noutperforms Large Model with Function-level Completions')
    plt.ylabel('Reduction in Carbon Emissions')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_yticks))
    plt.tight_layout()
    plt.savefig('output/reduction-carbon-emissions-if-small-models-outperforms-both', dpi=500)
    
                   

def test_results_figures():
    problems = DataProcessor.load('../../data/processed-dataset.jsonl')

    pass_results_small = collections.defaultdict(lambda: collections.defaultdict(list))
    pass_results_large = collections.defaultdict(lambda: collections.defaultdict(list))

    comp_results_small = collections.defaultdict(lambda: collections.defaultdict(list))
    comp_results_large = collections.defaultdict(lambda: collections.defaultdict(list))

    for _type in ['func_small', 'func_large']:
        for problem_id in map(str, range(1140)):
            if should_skip(problem_id):
                continue

            problem = problems[int(problem_id)]

            with open(glob.glob(f'{TEST_RESULTS_FOLDER}/{_type}/{problem_id}/*')[0], 'r') as f:
                func_test_results = [CodeVerificationResult(**json.loads(c)) for c in f.readlines()]

            pass_results = [r.tests_passed for r in func_test_results]
            comp_results = [r.compilation_passed for r in func_test_results]

            if len(problem.line_prompts) >= 10:
                pass_results_large[_type][1.0].append(sum(pass_results) / len(pass_results))
                comp_results_large[_type][1.0].append(sum(comp_results) / len(comp_results))
            else:
                pass_results_small[_type][1.0].append(sum(pass_results) / len(pass_results))
                comp_results_small[_type][1.0].append(sum(comp_results) / len(comp_results))

    bins_small = 5
    bins_large = 10

    bin_size_small = 1.0 / bins_small
    bin_size_large = 1.0 / bins_large

    for _type in ['line_small', 'line_large']:
        for problem_id in map(str, range(1140)):
            if should_skip(problem_id):
                continue

            problem = problems[int(problem_id)]
            num_prompts = len(problem.line_prompts)
            bin_size = bin_size_large if num_prompts >= 10 else bin_size_small

            with open(f'{MCS_RESULTS_FOLDER}/{_type}/{problem_id}.jsonl') as f:
                mcs_results = []

                for content in f.readlines():
                    mcs_result = MCSResult(**json.loads(content))
                    mcs_result.result = CodeVerificationResult(**mcs_result.result)
                    mcs_results.append(mcs_result)

                pass_results = collections.defaultdict(list)
                comp_results = collections.defaultdict(list)

                for mcs_result in mcs_results:
                    percentage = len(mcs_result.selected_lines) / max(num_prompts - 1, 1)
                    bin = round(math.ceil(percentage / bin_size) * bin_size, 1)
                    
                    pass_results[bin].append(mcs_result.result.tests_passed)
                    comp_results[bin].append(mcs_result.result.compilation_passed)

                for bin, results in pass_results.items():
                    (pass_results_large if num_prompts >= 10 else pass_results_small)[_type][bin].append(sum(results) / len(results))

                for bin, results in comp_results.items():
                    (comp_results_large if num_prompts >= 10 else comp_results_small)[_type][bin].append(sum(results) / len(results))

    def plot(results, title, file_name):
        data = []
        bins = sorted(results['line_small'].keys())

        for bin, pass_percentages in results['line_small'].items():
            data.extend([(bin, perc, 'Small Model') for perc in pass_percentages])

        for bin, pass_percentages in results['line_large'].items():
            data.extend([(bin, perc, 'Large Model') for perc in pass_percentages])

        df = pd.DataFrame(data, columns=['Fraction of Generated Lines', 'Test Pass Accuracy', 'Group'])

        plt.figure(figsize=(12, 4))
        sns.boxplot(x='Fraction of Generated Lines', y='Test Pass Accuracy', hue='Group', data=df, order=bins, showfliers=False, width=0.66)

        plt.title(title)
        plt.legend(title=None, fontsize='small', loc='lower left')
        plt.tight_layout()
        plt.savefig(file_name, dpi=500)
        plt.cla()

    plot(pass_results_small, 'Test Pass Accuracy per Fraction of Line-level Completions (Max Solution Length < 10)', 'output/test-pass-accuracy-per-fraction-line-completions-max-length')
    plot(pass_results_large, 'Test Pass Accuracy per Fraction of Line-level Completions', 'output/test-pass-accuracy-per-fraction-line-completions')

    data_small = []
    data_small.extend([(perc, 'Function-level, Small Model') for perc in pass_results_small['func_small'][1.0]])
    data_small.extend([(perc, 'Line-level, Small Model') for perc in pass_results_small['line_small'][1.0]])
    data_small.extend([(perc, 'Function-level, Large Model') for perc in pass_results_small['func_large'][1.0]])
    data_small.extend([(perc, 'Line-level, Large Model') for perc in pass_results_small['line_large'][1.0]])

    data_large = []
    data_large.extend([(perc, 'Function-level, Small Model') for perc in pass_results_large['func_small'][1.0]])
    data_large.extend([(perc, 'Line-level, Small Model') for perc in pass_results_large['line_small'][1.0]])
    data_large.extend([(perc, 'Function-level, Large Model') for perc in pass_results_large['func_large'][1.0]])
    data_large.extend([(perc, 'Line-level, Large Model') for perc in pass_results_large['line_large'][1.0]])

    df_small = pd.DataFrame(data_small, columns=['Test Pass Accuracy', 'Group'])
    df_large = pd.DataFrame(data_large, columns=['Test Pass Accuracy', 'Group'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(ax=axes[0], x='Group', y='Test Pass Accuracy', hue='Group', data=df_small, showfliers=False, width=0.5)
    sns.boxplot(ax=axes[1], x='Group', y='Test Pass Accuracy', hue='Group', data=df_large, showfliers=False, width=0.5)

    for ax in axes:
        ax.set_xlabel('')
        ax.set_xticks([])

    axes[0].set_title('Max Solution Length < 10')
    axes[1].set_title('Max Solution Length ≥ 10')

    import matplotlib.patches as mpatches

    unique_groups = df_small['Group'].unique()  # Groups should be the same for both plots
    colors = sns.color_palette(n_colors=len(unique_groups))  # Use seaborn color palette
    labels = unique_groups
    handles = [mpatches.Patch(facecolor=colors[i], label=labels[i], edgecolor="#434343", linewidth=1) for i in range(len(unique_groups))]

    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Test Pass Accuracy for Function-level and Line-level Completions (Fraction Generated Lines = 1.0)')
    
    plt.tight_layout(rect=[0, 0.125, 1, 1.025])
    plt.savefig('output/test-pass-accuracy-line-level-vs-function-level', dpi=500)

def cumulative_distribution_generation_time():
    energy_results = collections.defaultdict(lambda: collections.defaultdict(list))
    time_results = collections.defaultdict(lambda: collections.defaultdict(list))

    for _type in ['func_small', 'func_large', 'line_small', 'line_large']:
        for problem_id in map(str, range(1140)):
            if should_skip(problem_id):
                continue

            with open(f'{RESULTS_FOLDER}/{problem_id}/{_type}.jsonl') as f:
                for content in f.readlines():
                    result = GenerationResult(**json.loads(content))
                    result.metrics = GenerationMetrics(**result.metrics)

                    energy_results[_type][problem_id].append(result.metrics.energy)
                    time_results[_type][problem_id].append(result.metrics.time)

    func_small_energies = [sum(x) for x in energy_results['func_small'].values()]
    func_large_energies = [sum(x) for x in energy_results['func_large'].values()]
    line_small_energies = [sum(x) for x in energy_results['line_small'].values()]
    line_large_energies = [sum(x) for x in energy_results['line_large'].values()]

    data = []

    data.extend(zip(func_small_energies, ['Function-level, Small Model'] * len(func_small_energies)))
    data.extend(zip(line_small_energies, ['Line-level, Small Model'] * len(line_small_energies)))
    data.extend(zip(func_large_energies, ['Function-level, Large Model'] * len(func_large_energies)))
    data.extend(zip(line_large_energies, ['Line-level, Large Model'] * len(line_large_energies)))

    df = pd.DataFrame(data, columns=['Energy', 'Group'])
    df['Normalised Energy'] = df.groupby('Group')['Energy'].transform(lambda x: x / x.max())

    plt.figure(figsize=(8,5))

    sns.ecdfplot(data=df, y='Normalised Energy', hue='Group', stat='proportion')

    plt.xlim(-0.0025, 1.0025)
    plt.ylim(-0.005, 1.005)
    plt.xlabel('Normalised Energy')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution of Normalised Energy Consumption')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/cumulative-distribution-normalised-energy-consumption', dpi=500)
    plt.cla()

    func_small_times = [sum(x) for x in time_results['func_small'].values()]
    func_large_times = [sum(x) for x in time_results['func_large'].values()]
    line_small_times = [sum(x) for x in time_results['line_small'].values()]
    line_large_times = [sum(x) for x in time_results['line_large'].values()]

    data = []

    data.extend(zip(func_small_times, ['Function-level, Small Model'] * len(func_small_times)))
    data.extend(zip(line_small_times, ['Line-level, Small Model'] * len(line_small_times)))
    data.extend(zip(func_large_times, ['Function-level, Large Model'] * len(func_large_times)))
    data.extend(zip(line_large_times, ['Line-level, Large Model'] * len(line_large_times)))

    df = pd.DataFrame(data, columns=['Time', 'Group'])
    df['Normalised Time'] = df.groupby('Group')['Time'].transform(lambda x: x / x.max())

    plt.figure(figsize=(8,5))

    sns.ecdfplot(data=df, y='Normalised Time', hue='Group', stat='proportion')

    plt.xlim(-0.0025, 1.0025)
    plt.ylim(-0.005, 1.005)
    plt.xlabel('Normalised Time')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution of Normalised Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/cumulative-distribution-normalised-time', dpi=500)
    plt.cla()


def wasted_tokens_incorrect_suggestions():
    problems = DataProcessor.load('../../data/processed-dataset.jsonl')

    wasted_energies_lists = collections.defaultdict(list)

    for func_type, line_type in [('func_large', 'line_large'), ('func_small', 'line_small')]:
        func_large_wasted_energy = []
        line_wasted_energies = collections.defaultdict(list)

        for problem_id in map(str, range(1140)):
            if should_skip(problem_id):
                continue

            problem = problems[int(problem_id)]

            """
            FUNCTION-LEVEL
            """
            with open(f'{RESULTS_FOLDER}/{problem_id}/{func_type}.jsonl') as f:
                func_large_metrics = [GenerationMetrics(**GenerationResult(**json.loads(c)).metrics) for c in f.readlines()]

            with open(glob.glob(f'{TEST_RESULTS_FOLDER}/{func_type}/{problem_id}/*')[0], 'r') as f:
                func_large_test_results = [CodeVerificationResult(**json.loads(c)).tests_passed for c in f.readlines()]

            func_large = list(zip(func_large_test_results, func_large_metrics))

            func_large_total_energy = sum(x[1].energy for x in func_large)
            func_large_wrong_energy = sum(x[1].energy for x in func_large if not x[0])

            label = 'Function-level, Large Model' if 'large' in func_type else 'Function-level, Small Model'

            #wasted_energies_lists[label].append(func_large_wrong_energy / func_large_total_energy)
            #wasted_energies_lists[label].extend([ x[1].energy for x in func_large if not x[0] ])

            energy_list = [x[1].energy for x in func_large if not x[0]]
            if not energy_list:
                continue

            wasted_energies_lists[label].append(sum(energy_list) / len(energy_list))

            """
            LINE-LEVEL
            """
            energy_per_line = collections.defaultdict(list)

            with open(f'{RESULTS_FOLDER}/{problem_id}/{line_type}.jsonl') as f:
                for content in f.readlines():
                    result = GenerationResult(**json.loads(content))
                    result.outputs = [GenerationOutput(**o) for o in result.outputs]
                    result.metrics = GenerationMetrics(**result.metrics)

                    for line_prompt, output in zip(problem.line_prompts, result.outputs):
                        energy_per_line[line_prompt.line_index].append(len(output.tokens) * result.metrics.energy_per_token)

            energy_per_line = { i: sum(xs)/len(xs) for i, xs in energy_per_line.items() }

            line_large = collections.defaultdict(list)

            with open(f'{MCS_RESULTS_FOLDER}/{line_type}/{problem_id}.jsonl') as f:
                for content in f.readlines():
                    mcs_result = MCSResult(**json.loads(content))
                    mcs_result.result = CodeVerificationResult(**mcs_result.result)

                    total_energy = sum([energy_per_line[l] if l in energy_per_line else 0 for l in mcs_result.selected_lines])

                    line_large[len(mcs_result.selected_lines)].append((mcs_result.result.tests_passed, total_energy))

            total_lines = max(len(problem.line_prompts) - 1, 1)
            bin_size = 0.2

            for num_lines, energies in sorted(line_large.items()):
                bin = round(max(bin_size * round((num_lines / total_lines) / bin_size), bin_size), 1)

                if bin != 1.0:
                    continue

                label = 'Line-level, Large Model' if 'large' in line_type else 'Line-level, Small Model'

                line_large_total_energy = sum(x[1] for x in energies)
                line_large_wrong_energy = sum(x[1] for x in energies if not x[0])

                #wasted_energies_lists[label].append(line_large_wrong_energy / line_large_total_energy)
                #wasted_energies_lists[label].extend([ x[1] for x in energies if not x[0] ])

                energy_list = [x[1] for x in energies if not x[0]]
                if not energy_list:
                    continue

                wasted_energies_lists[label].append(sum(energy_list) / len(energy_list))

    import scipy.stats as stats

    def calculate_p_value(list1, list2):
        t_stat, p_value = stats.ttest_ind(list1, list2)
        return t_stat, p_value

    def calculate_p_value_nonparametric(list1, list2):
        u_stat, p_value = stats.mannwhitneyu(list1, list2, alternative='two-sided')
        return u_stat, p_value

    def is_normal_distribution(data, alpha=0.05):
        stat, p_value = stats.shapiro(data)
        return p_value > alpha, p_value

    print('Normal Distribution')
    print('Function-level, Small Model', is_normal_distribution(wasted_energies_lists['Function-level, Small Model']))
    print('Function-level, Large Model', is_normal_distribution(wasted_energies_lists['Function-level, Large Model']))
    print('Line-level, Small Model', is_normal_distribution(wasted_energies_lists['Line-level, Small Model']))
    print('Line-level, Large Model', is_normal_distribution(wasted_energies_lists['Line-level, Large Model']))
    print()

    print('U-stat, P-value')
    print('Small Model, function-level vs line-level', calculate_p_value_nonparametric(wasted_energies_lists['Function-level, Small Model'], wasted_energies_lists['Line-level, Small Model']))
    print('Large Model, function-level vs line-level', calculate_p_value_nonparametric(wasted_energies_lists['Function-level, Large Model'], wasted_energies_lists['Line-level, Large Model']))
    print('Function-level, small model vs large model', calculate_p_value_nonparametric(wasted_energies_lists['Function-level, Small Model'], wasted_energies_lists['Function-level, Large Model']))
    print('Line-level, small model vs large model', calculate_p_value_nonparametric(wasted_energies_lists['Line-level, Small Model'], wasted_energies_lists['Line-level, Large Model']))

    data = []
    for category, values in wasted_energies_lists.items():
        for value in values:
            data.append({'Category': category, 'Excess Tokens': value})

    df = pd.DataFrame(data)

    # Create subsets for the second graph
    line_level_data = df[df['Category'].isin(['Line-level, Small Model', 'Line-level, Large Model'])]

    # Create subplots
    plt.figure(figsize=(8,5))

    _palette = sns.color_palette()
    palette = [_palette[2], _palette[3], _palette[0], _palette[1]]

    sns.boxplot(x='Category', y='Excess Tokens', hue='Category', data=df, width=0.33, showfliers=False, palette=palette)

    plt.ylabel('Average Wasted Energy Consumption')
    plt.xlabel(None)
    plt.xticks([])
    plt.legend(labels=wasted_energies_lists.keys())
    plt.title('Average Wasted Energy Consumption per Problem due to Incorrect Suggestions')
    plt.tight_layout()
    plt.savefig('output/wasted-energy-incorrect-suggestions-average', dpi=500)

def boxplot_median_carbon_emission_reduction():
    data = {"Model": ["Small Model"]*3 + ["Large Model"]*3, "Value": [9, 16, 31, 2.5, 4.5, 9]}
    palette = sns.color_palette()

    ax = sns.boxplot(x="Model", y="Value", data=data, palette=palette, width=0.5)

    import matplotlib.patches as mpatches

    handles = [mpatches.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(2)]
    labels = ["Small Model", "Large Model"]
    plt.legend(handles, labels, title="Model Size")

    plt.xticks([])
    plt.xlabel(None)
    plt.title("Carbon Emission Reduction when Line-level Completions\noutperform Function-level Completions")
    plt.ylabel("Carbon Emission Reduction")
    plt.tight_layout()
    plt.savefig('output/boxplot-carbon-emission-reduction-llc-vs-flc', dpi=500)

if __name__ == '__main__':
    wasted_tokens_incorrect_suggestions()
    #test_results_figures()
    #wasted_tokens_incorrect_suggestions()
    
    #plot_small_outperforms_large()
    #fraction_where_small_model_outperforms_large_model()
    #plot_test_pass_accuracy()
    #test_pass_accuracy_line_level()
    #excess_tokens()
    #cumulative_distribution_generation_time()
    #test_results_figures()

    #wrong_solutions()

    #excess_tokens()
    #plt.cla()
    #boxplot_median_carbon_emission_reduction()
    #plot_small_outperforms_large()
    #plot_test_pass_accuracy()
    #plot_test_pass_accuracyreduction_carbon_emissions()

    #compress(LINE_SMALL_FOLDER, COMPRESSED_LINE_SMALL_FOLDER)
    #compress(LINE_LARGE_FOLDER, COMPRESSED_LINE_LARGE_FOLDER)

    #import matplotlib.pyplot as plt

    #load(COMPRESSED_LINE_SMALL_FOLDER)
    #load(COMPRESSED_LINE_LARGE_FOLDER, ax[1])

    #plt.show()
