import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import glob, json, pprint, re
import plotly.graph_objects as go

from classes import CodeVerificationResult, GenerationMetrics, GenerationOutput, GenerationResult
from collections import Counter, defaultdict
from data_processor import DataProcessor

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()
sns.color_palette("colorblind")

RESULTS_DIR = '/Users/thijsnulle/Documents/Git/msc-experiment/results'
TEST_RESULTS_DIR = '/Users/thijsnulle/Documents/Git/msc-experiment/test_results'
SKIPPED_PROBLEMS_FILE = '/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt'

def load_results(file_name):
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

def load_test_results(file_name, results, problems):
    skipped_problem_ids = []
    with open(SKIPPED_PROBLEMS_FILE) as f:
        for line in f.readlines():
            skipped_problem_ids.append(line.split(' - ')[0])

    output = []

    for file_name in glob.glob(f'{TEST_RESULTS_DIR}/{file_name}/**/*.jsonl'):
        problem_id = re.search(r'/(\d+)/', file_name)[1]
        line_index = int(re.search(r'(\d+)\.jsonl', file_name)[1])

        problem = problems[int(problem_id)]
        output_index = list(map(lambda x: x.line_index == line_index, problem.line_prompts)).index(True)

        if problem_id in skipped_problem_ids:
            continue

        generation_results = results[problem_id]
        generation_results_counter = Counter([ x.outputs[output_index].text for x in generation_results ])

        with open(file_name, 'r') as f:
            verification_results = [ CodeVerificationResult(**json.loads(x)) for x in f.readlines() ]

            for result in verification_results:
                output.extend([result] * generation_results_counter[result.code])

    return output

def power_law_distribution_unique_solutions():
    unique_solutions_counter = Counter()

    for file in glob.glob('data/power-law-distribution/**/*.txt'):
        with open(file) as f:
            num_unique_solutions = len(set([x for x in f.read().splitlines() if x.strip()]))
            
            if num_unique_solutions > 0:
                unique_solutions_counter[num_unique_solutions] += 1

    fig, ax = plt.subplots(figsize=(8,4))

    sns.histplot(ax=ax, data=list(unique_solutions_counter.elements()), bins=100, stat='proportion')

    plt.xlim(xmin=1)
    plt.title('Distribution of Total Unique Generations per Line')
    plt.xlabel('Number of Unique Generations')
    plt.tight_layout()
    plt.savefig('output/distribution-unique-line-generations')

    fig, ax = plt.subplots(figsize=(12,4))

    sns.histplot(ax=ax, data=list(unique_solutions_counter.elements()), bins=100, stat='proportion')

    plt.xlim(xmin=1)
    plt.title('Distribution of Total Unique Generations per Line')
    plt.xlabel('Number of Unique Generations')
    plt.tight_layout()
    plt.savefig('output/distribution-unique-line-generations-wide')

def sankey_diagram_test_results(small_results, large_results):
    def plot(results, file_addition):
        labels = [
            'Total Solutions',
            'Passed Compilation',
            'Did Not Compile',
            'Passed Tests',
            'Did Not Pass Tests',
        ]

        sources = []
        targets = []
        values = []

        compilation_passed = sum(x.compilation_passed for x in results)
        compilation_not_passed = len(results) - compilation_passed

        print(file_addition, compilation_passed, compilation_not_passed)

        values.extend([compilation_passed, compilation_not_passed])
        sources.extend([0, 0])
        targets.extend([1, 2])

        compilation_not_passed_results = [x for x in results if not x.compilation_passed]
        compilation_not_passed_errors = Counter([x.error for x in compilation_not_passed_results])

        for error, count in compilation_not_passed_errors.most_common():
            if not re.match('^\w+Error$', error) or count < 50:
                remaining += count
                continue

            labels.append(f'{error} ({count})')
            values.append(count)
            sources.append(2)
            targets.append(len(labels) - 1)
        
        tests_passed = sum(x.tests_passed for x in results)
        tests_not_passed = compilation_passed - tests_passed

        values.extend([tests_passed, tests_not_passed])
        sources.extend([1, 1])
        targets.extend([3, 4])

        tests_not_passed_results = [x for x in results if x.compilation_passed and not x.tests_passed]
        tests_not_passed_errors = Counter([x.error for x in tests_not_passed_results])

        remaining = 0

        for error, count in tests_not_passed_errors.most_common():
            if not re.match('^\w+Error$', error) or count < 50:
                remaining += count
                continue

            labels.append(f'{error} ({count})')
            values.append(count)
            sources.append(4)
            targets.append(len(labels) - 1)

        labels.append(f'Other ({remaining})')
        values.append(remaining)
        sources.append(4)
        targets.append(len(labels) - 1)

        labels[0] = f'{labels[0]} ({len(results)})'
        labels[1] = f'{labels[1]} ({compilation_passed})'
        labels[2] = f'{labels[2]} ({compilation_not_passed})'
        labels[3] = f'{labels[3]} ({tests_passed})'
        labels[4] = f'{labels[4]} ({tests_not_passed})'

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=12,
                thickness=30,
                label=labels,
                x=[0, 0.25, 0.5, 1, 0.5],
                y=[0, 0,    0.85, 0, 0.66],
            ),
            link = dict(
                source=sources,
                target=targets,
                value=values,
            )),
        ])

        fig.update_layout(
            template='seaborn',
            paper_bgcolor='#eaeaf2',
            font_size=14,
            width=1200,
            height=600,
            margin=dict(l=8, r=8, t=8, b=8),
        )

        fig.write_image(f'output/test-results-flow-sankey-diagram-{file_addition}.png')

    plot(small_results, file_addition='line-small')
    plot(large_results, file_addition='line-large')

        
if __name__ == '__main__':
    problems = DataProcessor.load(input_file_path='/Users/thijsnulle/Documents/Git/msc-experiment/data/test-dataset.jsonl')

    small_results = load_results('line_small')
    large_results = load_results('line_large')

    small_test_results = load_test_results('line_small', small_results, problems)
    large_test_results = load_test_results('line_large', large_results, problems)

    sankey_diagram_test_results(small_test_results, large_test_results)

    #line_small = load_results('line_small')
    #power_law_distribution_unique_solutions()

