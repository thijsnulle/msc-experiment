import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast
import contextlib
import dataclasses
import gc
import glob
import io
import json
import multiprocessing
import os
import psutil
import re
import resource
import signal
import threading
import time
import unittest

from classes import CodeVerificationResult, DataclassJSONEncoder, InputPrompt, GenerationMetrics, GenerationOutput, GenerationResult
from collections import defaultdict
from data_processor import DataProcessor
from pathlib import Path
from pathlib import Path
from typing import Optional

RESULTS_DIR = '/Users/thijsnulle/Documents/Git/msc-experiment/results'

SKIPPED_PROBLEMS_FILE = Path('/Users/thijsnulle/Documents/Git/msc-experiment/tests/skipped_problems.txt')
VERIFICATIONS_DIR_PATH = Path('/Users/thijsnulle/Documents/Git/msc-experiment/test_results')

def test_handler(signum, frame):
    raise TimeoutError()

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


class CodeVerifier:
    def __init__(self):
        self.cache = defaultdict(lambda: defaultdict(set))

    def verify(self, problem, generation_results, dataset_type, line_level):
        self._load_cache(problem, dataset_type)

        with open(SKIPPED_PROBLEMS_FILE, 'a+'):
            pass

        with open(SKIPPED_PROBLEMS_FILE, 'r') as f:
            for line in f.readlines():
                if line.startswith(f'{problem.id} '):
                    return

        test_lines = problem.reference.test_code.splitlines()
        test_lines.insert(1, 'from unittest.mock import patch\npatch("matplotlib.pyplot.show").start()')
        problem.reference.test_code = '\n'.join(test_lines)

        if not bool(self.cache):
            # if tests of problem take >2 seconds, skip for now.
            signal.signal(signal.SIGALRM, test_handler)
            signal.alarm(2)

            try:
                with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                    verification_result = self._verify_line(problem, InputPrompt('', 0, 100), '')
            except:
                verification_result = None

            signal.alarm(0)

            if verification_result and verification_result.error:
                print('Verification error:', verification_result.error)

            if not verification_result or verification_result.error == 'TimeoutError' or verification_result.time > 2.0:
                print(f'Problem {problem.id} --- Skipped')

                with open(SKIPPED_PROBLEMS_FILE, 'a+') as f:
                    f.write(problem.id)
                    f.write(' - ')
                    f.write(json.dumps(verification_result, cls=DataclassJSONEncoder)) 
                    f.write('\n')

                return

        code_lines = [x for x in problem.reference.code.splitlines() if x.strip()]

        problem_path = VERIFICATIONS_DIR_PATH / dataset_type / problem.id
        problem_path.mkdir(parents=True, exist_ok=True)

        cache_hits = 0
        non_cache_hits = 0

        for cache in self.cache.values():
            non_cache_hits += len(cache)

        current_cache_size = non_cache_hits

        for result in generation_results:
            for i, output in enumerate(result.outputs):
                #if psutil.Process().memory_info().rss / 1024**2 > 2000:
                #    gc.collect()

                # TODO: perform actual data processing to remove this.

                line_prompt = problem.line_prompts[i]
                func_prompt = problem.func_prompt

                code = output.text if '<|endoftext|>' not in output.text else output.text.split('<|endoftext|>')[0]

                if not line_level:
                    if code_split := re.split(r'\n\n\S', code):
                        code = code_split[0]

                if code in self.cache[line_prompt.line_index]:
                    cache_hits += 1
                    continue
                else:
                    non_cache_hits += 1

                #print(psutil.Process().memory_info().rss / 1024**2, 'MB')
                print(problem.id, line_prompt.line_index, repr(code))

                with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                    if line_level:
                        verification_result = self._verify_line(problem, line_prompt, code)
                    else:
                        verification_result = self._verify_func(problem, func_prompt, code)

                self.cache[line_prompt.line_index][code] = verification_result

                with open(problem_path / f'{line_prompt.line_index}.jsonl', 'a+') as f:
                    f.write(json.dumps(verification_result, cls=DataclassJSONEncoder))
                    f.write('\n')

        tests_passed = 0
        tests_total = 0

        for _, line_index_values in self.cache.items():
            for result in line_index_values.values():
                tests_total += 1
                tests_passed += 1 if result.tests_passed else 0

        cache_hits -= current_cache_size # To offset from the already existing elements in the cache that were doubled counted

        print(f'Problem {problem.id}\t- Cache Hit %: {cache_hits/(cache_hits+non_cache_hits):.2f},\tTest Pass %: {tests_passed/tests_total:.2f}, \tMemory Usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB') 


    def _verify_line(self, problem, line_prompt, code):
        t0 = time.time()

        line_index = line_prompt.line_index
        char_index = line_prompt.char_index

        compiled_code, did_compile, error = self._compile_line(problem, line_index, char_index, code)

        if not did_compile:
            return CodeVerificationResult(code=code, compilation_passed=False, error=error, time=time.time() - t0)

        tests_passed, error = self._test(compiled_code)

        if not tests_passed:
            return CodeVerificationResult(code=code, compilation_passed=True, tests_passed=False, error=error, time=time.time() - t0)

        return CodeVerificationResult(code=code, compilation_passed=True, tests_passed=True, time=time.time() - t0)

    def _compile_line(self, problem, line_index, char_index, code):
        """
        returns: code, compilation_passed, error
        """
        solution_lines = problem.reference.complete_code.splitlines()
        solution_lines[line_index] = solution_lines[line_index][:char_index] + code
        solution_lines.extend(problem.reference.test_code.splitlines())

        solution = '\n'.join(solution_lines)

        try:
            abstract_syntax_tree = ast.parse(solution)
            code = compile(abstract_syntax_tree, filename='<string>', mode='exec')
        except Exception as e:
            return None, False, type(e).__name__

        return code, True, None

    def _verify_func(self, problem, func_prompt, code):
        t0 = time.time()

        line_index = func_prompt.line_index
        char_index = func_prompt.char_index

        compiled_code, did_compile, error = self._compile_func(problem, line_index, char_index, code)

        if not did_compile:
            return CodeVerificationResult(code=code, compilation_passed=False, error=error, time=time.time() - t0)

        tests_passed, error = self._test(compiled_code)

        if not tests_passed:
            return CodeVerificationResult(code=code, compilation_passed=True, tests_passed=False, error=error, time=time.time() - t0)

        return CodeVerificationResult(code=code, compilation_passed=True, tests_passed=True, time=time.time() - t0)

    def _compile_func(self, problem, line_index, char_index, code):
        """
        returns: code, compilation_passed, error
        """
        solution_lines = problem.reference.complete_code.splitlines()[:line_index]
        solution_lines.append(f'    {code.splitlines()[0]}')
        solution_lines.extend(code.splitlines()[1:])
        solution_lines.extend(problem.reference.test_code.splitlines())

        solution = '\n'.join(solution_lines)

        try:
            abstract_syntax_tree = ast.parse(solution)
            code = compile(abstract_syntax_tree, filename='<string>', mode='exec')
        except Exception as e:
            return None, False, type(e).__name__

        return code, True, None

    def _test(self, code):
        """
        returns: tests_passed, exception, time
        """

        buffer = io.StringIO()
        context = {}

        signal.signal(signal.SIGALRM, test_handler)
        signal.alarm(1)

        try:
            exec(code, context)

            runner = unittest.TextTestRunner(stream=buffer, failfast=True, verbosity=0)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(context['TestCases'])

            test_result = runner.run(suite)
        except Exception as e:
            print('Exception while testing:', e)
            return False, str(e)
        finally:
            signal.alarm(0)

        if test_result.errors:
            exception = test_result.errors[0][1].splitlines()[-1].split(':')[0]

            return False, exception

        return True, None

    def _load_cache(self, problem, dataset_type):
        for line_cache in self.cache.values():
            for line_cache_set in line_cache.values():
                del line_cache_set
                    
            line_cache.clear()

        self.cache.clear()

        problem_path = VERIFICATIONS_DIR_PATH / dataset_type / problem.id

        if not problem_path.exists():
            return

        for problem_file in glob.glob(f'{problem_path}/*.jsonl'):
            line_index = int(re.search('(\d+)\.jsonl', problem_file)[1])

            with open(problem_file, 'r') as f:
                for verification_result_str in f.readlines():
                    result = CodeVerificationResult(**json.loads(verification_result_str))

                    self.cache[line_index][result.code] = result
        
if __name__ == '__main__':
    code_verifier = CodeVerifier()

    #line_small = load_data('line_small')
    #line_large = load_data('line_large')

    problems = DataProcessor.load(input_file_path='../data/test-dataset.jsonl')

    #for problem_id, generation_results in line_large.items():
    #    problem = problems[int(problem_id)]       
    #    result = code_verifier.verify(problem, generation_results, dataset_type='line_large', line_level=True)

    #func_small = load_data('func_small')
    func_large = load_data('func_large')

    i = 0

    for problem_id, generation_results in func_large.items():
        problem = problems[int(problem_id)]
        result = code_verifier.verify(problem, generation_results, dataset_type='func_large', line_level=False)

