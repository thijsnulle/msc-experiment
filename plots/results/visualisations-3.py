import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import collections, json, re

from classes import CodeVerificationResult, MCSResult
from data_processor import DataProcessor

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import LabelEncoder

sns.set_theme()
sns.color_palette("colorblind")

MCS_FOLDER = '/Users/thijsnulle/Documents/Git/msc-experiment/mcs_results'

def should_skip(problem_id):
    with open('/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt', 'r') as f:
        return any(l.startswith(f'{problem_id} - ') for l in f.readlines())

def mean_median_per_statement():
    problems = DataProcessor.load('/Users/thijsnulle/Documents/Git/msc-experiment/data/processed-dataset.jsonl')

    statement_collection = collections.defaultdict(list)

    for problem in problems:
        if should_skip(problem.id):
            continue

        with open(f'{MCS_FOLDER}/line_large/{problem.id}.jsonl') as f:
            mcs_results = [MCSResult(**json.loads(c)) for c in f.readlines()]
            
            for r in mcs_results:
                r.result = CodeVerificationResult(**r.result)

        total_counter = collections.Counter(line for r in mcs_results for line in r.selected_lines)
        line_counter = collections.Counter(line for r in mcs_results if r.result.tests_passed for line in r.selected_lines)

        for prompt in problem.line_prompts:
            total = total_counter[prompt.line_index]
            count = line_counter[prompt.line_index]

            if not total:
                continue

            statement = prompt.prompt.splitlines()[-1].strip()
            statement = 'for' if statement.startswith('for') else statement.split()[-1]

            for x in '.,:(){}=':
                if statement.endswith(x) and not statement.endswith(('+=', '-=')):
                    statement = x

            if statement in [')', '}']:
                continue

            statement_collection[statement].append(count / total)

    data = []
    for key, values in statement_collection.items():
        for value in values:
            data.append({"Category": key, "Value": value})

    df = pd.DataFrame(data)

    stats_df = df.groupby("Category")["Value"].agg(["median", "mean", "count"])
    stats_df = stats_df[stats_df["count"] >= 5].drop(columns=["count"]).sort_values(by="median")

    stats_melted = stats_df.reset_index().melt(id_vars="Category", var_name="Statistic", value_name="Accuracy")

    plt.figure(figsize=(12, 5))
    sns.barplot(x="Category", y="Accuracy", hue="Statistic", data=stats_melted)

    plt.xticks(rotation=90)
    plt.xlabel("Trigger Point")  
    plt.ylabel("Accuracy")  
    plt.title("Mean and Median Test Pass Accuracies by Trigger Point")  
    plt.legend(title="Statistic")

    plt.tight_layout()
    plt.grid(True)
    plt.savefig('output/mean-median-per-statement', dpi=500)

def correlation():
    problems = DataProcessor.load('/Users/thijsnulle/Documents/Git/msc-experiment/data/processed-dataset.jsonl')

    test_results = collections.defaultdict(list)

    for problem in problems:
        if should_skip(problem.id):
            continue

        with open(f'{MCS_FOLDER}/line_small/{problem.id}.jsonl') as f:
            mcs_results = [MCSResult(**json.loads(c)) for c in f.readlines()]
            
            for r in mcs_results:
                r.result = CodeVerificationResult(**r.result)

        for result in mcs_results:
            test_results[len(problem.line_prompts)].append(result.result.tests_passed)

    true_frequencies = [sum(values) for values in test_results.values()]

    keys = list(test_results.keys())

    spearman_correlation, _ = spearmanr(keys, true_frequencies)

    print("Spearman", spearman_correlation)

def trigger_points_distribution():
    problems = DataProcessor.load('/Users/thijsnulle/Documents/Git/msc-experiment/data/processed-dataset.jsonl')

    statement_counter = collections.Counter()
    count = 0
    for problem in problems:
        if should_skip(problem.id):
            continue

        for prompt in problem.line_prompts:
            statement = prompt.prompt.splitlines()[-1].strip()
            statement = 'for' if statement.startswith('for') else statement.split()[-1]

            if statement == 'Ignor':
                continue

            for x in '.,:(){}=':
                if statement.endswith(x) and not statement.endswith(('+=', '-=')):
                    statement = x

            statement_counter[statement] += 1

    for statement, count in statement_counter.most_common():
        print(count, statement)

import re

def wrong_descriptions():
    problems = DataProcessor.load('/Users/thijsnulle/Documents/Git/msc-experiment/data/processed-dataset.jsonl')

    problems_without_raises_docstring = list(filter(lambda p: 'raise ' in p.reference.complete_code and not 'Raises:' in p.reference.complete_code, problems))
    problems_with_wrong_import = []
    problems_with_wrong_from_import = []

    for i, problem in enumerate(problems):
        imports = re.findall(r'^import (.*) as ', problem.reference.complete_code, flags=re.MULTILINE)
        from_imports = re.findall(r'^from (.*) import (.*)', problem.reference.complete_code, flags=re.MULTILINE)

        if not imports and not from_imports:
            continue

        requirements_text = re.split(r'\n\s*\n', re.search(r'Requirements:\n(\s+-? .*\n)+', problem.reference.complete_code).group(0))[0]
        requirements_list = [r.split(' - ')[1] if ' - ' in r else r.strip() for r in requirements_text.splitlines()[1:]]

        for i in imports:
            if not any(r.startswith(i) for r in requirements_list):
                problems_with_wrong_import.append(problem)
                break

        for lib, i in from_imports:
            i = f'{lib}.{i}'

            if i not in requirements_list:
                problems_with_wrong_from_import.append(problem)
        
    print('Problems with wrong import-statement:', len(problems_with_wrong_import))
    print('Problems with wrong `from` import-statement:', len(problems_with_wrong_from_import))
    print('Problems with `raise` but not in docstring:', len(problems_without_raises_docstring))
    
if __name__ == '__main__':
    #wrong_descriptions()
    trigger_points_distribution()

