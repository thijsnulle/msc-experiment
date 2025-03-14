import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import contextlib
import collections
import gc
import glob
import io
import json
import psutil
import random
import re
import signal
import time
import traceback
import unittest

from pprint import pprint

from collections import defaultdict
from classes import GenerationResult, GenerationOutput, GenerationMetrics
from classes import CodeVerificationResult, MCSResult, DataclassJSONEncoder
from data_processor import DataProcessor
from pathlib import Path

EXPERIMENT_RESULTS_PATH = Path('/Users/thijsnulle/Documents/Git/msc-experiment/results')
MCS_RESULTS_PATH = Path('/Users/thijsnulle/Documents/Git/msc-experiment/mcs_results')
TEST_RESULTS_PATH = Path('/Users/thijsnulle/Documents/Git/msc-experiment/test_results')

def test_handler(signum, frame):
    raise TimeoutError()

def load_data(file_name):
    output = defaultdict(list)

    for file in glob.glob(f'{EXPERIMENT_RESULTS_PATH}/**/{file_name}.jsonl'):
        problem_id = re.search(r'(\d+)', file)[0]

        with open(file) as f:
            for line in f.read().splitlines():
                result = GenerationResult(**json.loads(line))
                result.outputs = [GenerationOutput(**o) for o in result.outputs]
                result.metrics = GenerationMetrics(**result.metrics)

                output[problem_id].append(result)

    return output

class MonteCarloSimulation:
    def __init__(self, problem_id, data, dataset_type, K=10000):
        problems = DataProcessor.load(input_file_path='../data/test-dataset.jsonl')

        self.problem = problems[int(problem_id)]
        self.type = dataset_type

        with open(f'{MCS_RESULTS_PATH}/{self.type}/{self.problem.id}.jsonl', 'a+'):
            pass

        with open(f'{MCS_RESULTS_PATH}/{self.type}/{self.problem.id}.jsonl', 'r') as f:
            self.K = K - len(f.readlines())

        self._load_data(data)
        self._load_test_results()

    def start(self):
        process = psutil.Process(os.getpid())

        for k in range(self.K):
            if process.memory_info().rss > 6.0 * 1024**3:
                print('Exitting due to insufficient memory available.')
                os._exit(1)

            # select fraction 0 <= f <= 1 lines L_f, where |L_f| >= 2
            lines_to_substitute = random.randrange(1, len(self.problem.line_prompts)) \
                if len(self.problem.line_prompts) > 1 else len(self.problem.line_prompts)
            selected_line_prompts = random.sample(self.problem.line_prompts, lines_to_substitute)

            # substitute f * |S| lines
            replacements = []
            code = self.problem.reference.complete_code.splitlines()

            for prompt in selected_line_prompts:
                line_index, char_index = prompt.line_index, prompt.char_index

                replacement = random.choice(self.data[line_index])
                replacements.append((replacement, line_index))

                code[line_index] = code[line_index][:char_index] + replacement

            code = '\n'.join(code)

            # check if all pass compilation and tests, if not continue
            if code in self.test_results:
                mcs_result = self.test_results[code]
                mcs_result.selected_lines = [x.line_index for x in selected_line_prompts]

                self._save(mcs_result)

                continue

            # check for individual selected lines if they do not pass tests, if not continue
            if any(not self.line_results[i][r] for r, i in replacements):
                mcs_result = MCSResult(
                    selected_lines=[x.line_index for x in selected_line_prompts],
                    result=CodeVerificationResult(
                        code=code,
                        compilation_passed=True, # TODO
                        tests_passed=False,      # TODO
                        time=0.0,                # TODO
                        error=None,              # TODO
                    ),
                )

                self._save(mcs_result)

                continue

            t0 = time.time()
            tests_passed, error = self._test(code)

            mcs_result = MCSResult(
                selected_lines=[x.line_index for x in selected_line_prompts],
                result=CodeVerificationResult(
                    code=code,
                    compilation_passed=True,
                    tests_passed=tests_passed,
                    time=time.time() - t0,
                    error=error,
                ),
            )

            self.test_results[code] = mcs_result
            self._save(mcs_result)

    def _test(self, code):
        """
        returns: tests_passed, exception
        """
        buffer = io.StringIO()
        context = {}

        test_lines = ['from unittest.mock import patch\npatch("matplotlib.pyplot.show").start()']
        test_lines.extend(self.problem.reference.test_code.splitlines())
        test_lines.extend(code.splitlines())

        test_code = '\n'.join(test_lines)

        signal.signal(signal.SIGALRM, test_handler)
        signal.alarm(1)

        try:
            exec(test_code, context)

            runner = unittest.TextTestRunner(stream=buffer, failfast=True, verbosity=0)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(context['TestCases'])
            
            with contextlib.redirect_stderr(None), contextlib.redirect_stdout(None):
                test_result = runner.run(suite)
        except Exception as e:
            #print('Exception while testing:', e)
            return False, str(e)
        finally:
            signal.alarm(0)

        if test_result.errors:
            exception = test_result.errors[0][1].splitlines()[-1].split(':')[0]

            return False, exception

        return True, None

    def _save(self, result):
        with open(f'{MCS_RESULTS_PATH}/{self.type}/{self.problem.id}.jsonl', 'a+') as f:
            f.write(json.dumps(result, cls=DataclassJSONEncoder))
            f.write('\n')

    def _load_data(self, data):
        self.data = defaultdict(list)

        line_indexes = list(map(lambda p: p.line_index, self.problem.line_prompts))

        for data_point in data:
            for i, output in enumerate(data_point.outputs):
                text = output.text if '<|endoftext|>' not in output.text else output.text.split('<|endoftext|>')[0]

                self.data[line_indexes[i]].append(output.text)

    def _load_test_results(self):
        test_results_path = TEST_RESULTS_PATH / self.type / self.problem.id

        self.test_results = defaultdict(bool)
        self.line_results = defaultdict(lambda: defaultdict(bool))
        
        for file_name in glob.glob(f'{test_results_path}/**.jsonl'):
            line_index = int(re.search(r'(\d+)\.jsonl', file_name)[1])
            line_prompt = list(filter(lambda p: p.line_index == line_index, self.problem.line_prompts))[0]

            with open(file_name) as f:
                for content in f.readlines():
                    result = CodeVerificationResult(**json.loads(content))

                    self.line_results[line_index][result.code] = result.tests_passed

                    code = self.problem.reference.complete_code.splitlines()
                    code[line_index] = code[line_index][:line_prompt.char_index] + result.code
                    result.code = '\n'.join(code)

                    self.test_results[result.code] = MCSResult(selected_lines=[line_index], result=result)

        with open(f'{MCS_RESULTS_PATH}/{self.type}/{self.problem.id}.jsonl') as f:
            for line in f.read().splitlines():
                mcs_result = MCSResult(**json.loads(line))
                mcs_result.result = CodeVerificationResult(**mcs_result.result)

                self.test_results[mcs_result.result.code] = mcs_result


def main():
    try:
        line_small = load_data('line_small')

        for problem_id in [str(x) for x in range(1140)]:
            with open('/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt') as f:
                if any(line.split(' - ')[0] == problem_id for line in f.readlines()):
                    #print(f'Skipping problem {problem_id}')
                    continue

            MonteCarloSimulation(problem_id=problem_id, data=line_small[problem_id], dataset_type='line_small').start()

        line_large = load_data('line_large')

        for problem_id in [str(x) for x in range(1140)]:
            with open('/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt') as f:
                if any(line.split(' - ')[0] == problem_id for line in f.readlines()):
                    #print(f'Skipping problem {problem_id}')
                    continue

            MonteCarloSimulation(problem_id=problem_id, data=line_large[problem_id], dataset_type='line_large').start()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        #traceback.print_exc()
        #print("Restarting the script...")
        main()

if __name__ == '__main__':
    main()

