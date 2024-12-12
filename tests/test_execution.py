import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast, json, glob, re, signal, time, unittest
import matplotlib.pyplot as plt

from classes import GenerationResult, GenerationOutput, GenerationMetrics
from contextlib import redirect_stderr, redirect_stdout
from collections import defaultdict
from data_processor import DataProcessor
from io import StringIO

RESULTS_DIR = '/Users/thijs/Documents/Git/msc-experiment/results'

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

def compile(solution):
    try:
        ast.parse(solution)
        return True
    except:
        return False

def test_handler(signum, frame):
    raise RuntimeError()

def test(solution, test_code):
    buffer = StringIO()
    context = {}

    t0 = time.time()

    signal.signal(signal.SIGALRM, test_handler)
    signal.alarm(1)

    try:
        exec(f'{solution}\n{test_code}', context)
        
        runner = unittest.TextTestRunner(stream=buffer, failfast=True, verbosity=0)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(context['TestCases'])

        runner.run(suite)
    except Exception:
        return False, time.time() - t0

    return not buffer.getvalue().startswith('='), time.time() - t0

if __name__ == '__main__':
    line_small = load_data('line_small')
    line_large = load_data('line_large')

    problems = DataProcessor.load(input_file_path='data/test-dataset.jsonl')

    for problem in problems:
        if int(problem.id) not in [8, 11, 23, 26, 33]:
        #if int(problem.id) not in [23]:
            continue

        if problem.id not in line_small or problem.id not in line_large:
            continue

        print(f'Problem {problem.id}')

        total_solutions = 0
        total_executions = 0
        did_not_compile = 0
        did_not_pass_tests = 0
        test_execution_times = []

        correct_lines = defaultdict(set)

        with redirect_stdout(None), redirect_stderr(None):
            cached_test_results = defaultdict(dict)

            for dataset in [line_small]:
                for i, result in enumerate(dataset[problem.id]):
                    for j, prompt in enumerate(problem.line_prompts):
                        output = result.outputs[j]

                        line_index = prompt.line_index
                        char_index = prompt.char_index
                        
                        lines = problem.reference.complete_code.splitlines()
                        lines[line_index] = lines[line_index][:char_index] + output.text

                        solution = '\n'.join(lines)

                        total_solutions += 1

                        if not compile(solution):
                            did_not_compile += 1
                            continue

                        if output.text in cached_test_results[line_index]:
                            if not cached_test_results[line_index][output.text][0]:
                                did_not_pass_tests += 1

                            continue

                        tests_passed, t = test(solution, problem.reference.test_code)
                        total_executions += 1

                        cached_test_results[line_index][output.text] = (tests_passed, t)

                        test_execution_times.append(t)

                        if not tests_passed:
                            did_not_pass_tests += 1
                            continue

                        correct_lines[j].add((output.text, line_index, char_index))

        if len(correct_lines.values()) < len(problem.line_prompts):
            print('One of the lines is always wrong')

        """
        import itertools

        combinations = 1
        for c in correct_lines.values():
            combinations *= len(c)
        print(combinations)

        combinations = [list(c) for c in itertools.product(*correct_lines.values())]

        wrong = None

        for i, c in enumerate(combinations):
            solution = problem.reference.complete_code.splitlines()

            for text, line_index, char_index in c:
                solution[line_index] = solution[line_index][:char_index] + text

            solution = '\n'.join(solution)

            tests_passed, _ = test(solution, problem.reference.test_code)

            if not tests_passed:
                print(f'Tests have not passed {i}, {c}')

                if wrong == None:
                    wrong = set(c)
                else:
                    wrong = wrong & set(c)

        if wrong:
            print(problem.reference.complete_code)
            print()
            print(problem.reference.test_code)
            print()

            solution = problem.reference.complete_code.splitlines()

            for i, (text, line_index, char_index) in enumerate(list(wrong)):
                print(repr(solution[line_index]), repr(text))

                solution[line_index] = solution[line_index][:char_index] + text

            solution = '\n'.join(solution)

            tests_passed, _ = test(solution, problem.reference.test_code)

            print(tests_passed)


        print(f'Did not compile: {did_not_compile}/{total_solutions}')
        print(f'Did not pass tests: {did_not_pass_tests}/{total_solutions - did_not_compile}')
        print(f'Total executions: {total_executions}')
        print(f'Average test execution time: {sum(test_execution_times) / len(test_execution_times)}')
        print()
        """

    """
    line_small_fractions = [(1-l)*100 for l in line_small_fractions]
    line_large_fractions = [(1-l)*100 for l in line_large_fractions]

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme()
    sns.color_palette('colorblind')

    fig, ax = plt.subplots(figsize=(8,4))

    df = pd.DataFrame({'Value': line_small_fractions, 'Category': ['Small'] * len(line_small_fractions)})
    sns.violinplot(x='Category', y='Value', label='Small Model (1.5B)', data=df, ax=ax, inner=None)

    df = pd.DataFrame({'Value': line_large_fractions, 'Category': ['Large'] * len(line_large_fractions)})
    sns.violinplot(x='Category', y='Value', label='Large Model (9B)', data=df, ax=ax, inner=None)

    plt.legend()
    plt.xlabel(None)
    plt.ylabel('Percentage (%)')
    plt.title('Line-Level Compilation Percentage Distribution')
    plt.savefig('plots/results/output/line-level-compilation-distribution')
    """


