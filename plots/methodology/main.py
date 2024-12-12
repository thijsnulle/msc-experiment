import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import glob, json, re
import plotly.graph_objects as go

from classes import CodeVerificationResult, GenerationMetrics, GenerationOutput, GenerationResult
from collections import Counter, defaultdict

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()
sns.color_palette("colorblind")

RESULTS_DIR = '/Users/thijs/Documents/Git/msc-experiment/results'
TEST_RESULTS_DIR = '/Users/thijs/Documents/Git/msc-experiment/tests/results/line_small'

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

def load_test_results():
    output = []

    for file_name in glob.glob(f'{TEST_RESULTS_DIR}/**/*.jsonl'):
        with open(file_name, 'r') as f:
            output.extend([ CodeVerificationResult(**json.loads(x)) for x in f.readlines() ])

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

def sankey_diagram_test_results(results):
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

    values.extend([compilation_passed, compilation_not_passed])
    sources.extend([0, 0])
    targets.extend([1, 2])

    compilation_not_passed_results = [x for x in results if not x.compilation_passed]
    compilation_not_passed_errors = Counter([x.error for x in compilation_not_passed_results])

    for error, count in compilation_not_passed_errors.most_common():
        if not re.match('^\w+Error$', error) or count < 10:
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
        if not re.match('^\w+Error$', error) or count < 10:
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
            x=[0, 0.33, 0.33, 1, 0.5],
            y=[0, 0,    0.95,    0, 0.75],
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
        font_size=13,
        width=800,
        height=400,
        margin=dict(l=8, r=8, t=8, b=8),
    )

    fig.write_image('output/test-results-flow-sankey-diagram.png')

        
if __name__ == '__main__':
    #line_small = load_results('line_small')
    test_results = load_test_results()

    sankey_diagram_test_results(test_results)

    #power_law_distribution_unique_solutions()

